import atexit
import json
import os
import re
import uuid
import time
import threading
import ollama
from app.core.config import STM_WINDOW, SESSIONS_DIR, PROJECTS_DIR, MAX_TOOL_CALL_LOOPS
from app.core.database import ChromaMemoryDB
from app.utils.compat import _embed
from app.services.scraper_service import get_web_context
from app.services.pdf_service import ingest_pdf
from app.services.image_service import ImageService
from app.core.providers.manager import ProviderManager
from app.core.providers.base import ProviderError, CONTEXT_LENGTH_EXCEEDED, sanitize_error
from app.personalities.registry import PersonalityRegistry


class LocalDoremonMaster:
    def __init__(self):
        self.bot_name = "Doremon"
        self.db = ChromaMemoryDB()
        self.img_service = ImageService()
        self.provider_manager = ProviderManager()
        self.pending_approvals = {}
        self._approval_lock = threading.Lock()
        self._model_cache = None
        self._model_cache_time = 0
        self._model_cache_ttl = 30
        self._cancel_events: dict[str, threading.Event] = {}
        self._cancel_lock = threading.Lock()

        # User state tracking: user_id -> dict
        self.users_state = {}

        os.makedirs(SESSIONS_DIR, exist_ok=True)
        PersonalityRegistry.discover()
        atexit.register(self._shutdown)

    def _init_user(self, user_id: str):
        if user_id not in self.users_state:
            new_session_id = uuid.uuid4().hex
            self.users_state[user_id] = {
                "sessions": {new_session_id: {"messages": [], "attachments": []}},
                "active_session": new_session_id,
                "model": "llama3.2",
                "personality": "standard",
                "work_dir": "",
                "undo_point": None
            }
            self._load_user_sessions(user_id)

    def _load_user_sessions(self, user_id: str):
        user_dir = os.path.join(SESSIONS_DIR, user_id)
        os.makedirs(user_dir, exist_ok=True)
        for f in os.listdir(user_dir):
            if f.endswith(".json"):
                filepath = os.path.join(user_dir, f)
                try:
                    with open(filepath, "r", encoding="utf-8") as file:
                        data = json.load(file)
                        sid = data["id"]
                        self.users_state[user_id]["sessions"][sid] = {
                            "messages": data.get("messages", []),
                            "attachments": data.get("attachments", [])
                        }
                except Exception:
                    pass

    def _get_user_state(self, user_id: str):
        self._init_user(user_id)
        return self.users_state[user_id]

    def _get_stm(self, user_id: str, session_id: str = None) -> list:
        state = self._get_user_state(user_id)
        sid = session_id or state["active_session"]
        if sid not in state["sessions"]:
            state["sessions"][sid] = {"messages": [], "attachments": []}
        return state["sessions"][sid]["messages"]

    def get_active_session(self, user_id: str) -> str:
        return self._get_user_state(user_id)["active_session"]

    def get_current_model(self, user_id: str) -> str:
        return self._get_user_state(user_id)["model"]

    def get_active_personality(self, user_id: str):
        state = self._get_user_state(user_id)
        return PersonalityRegistry.get(state.get("personality", "standard"))

    def set_personality(self, user_id: str, personality_id: str) -> str:
        state = self._get_user_state(user_id)
        personality = PersonalityRegistry.get(personality_id)
        if not personality:
            return f"[ERROR] Personality '{personality_id}' not found"
        state["personality"] = personality_id
        return f"[PERSONALITY] Switched to '{personality.name}'"

    def get_work_dir(self, user_id: str) -> str:
        return self._get_user_state(user_id).get("work_dir", "")

    def set_work_dir(self, user_id: str, path: str) -> str:
        state = self._get_user_state(user_id)
        if path and not os.path.isdir(path):
            return f"[ERROR] Directory does not exist: {path}"
        state["work_dir"] = path
        if path:
            self.record_project_use(user_id, path)
            git_dir = os.path.join(path, ".git")
            if not os.path.isdir(git_dir):
                try:
                    import subprocess
                    subprocess.run(["git", "init"], cwd=path, capture_output=True, text=True, timeout=10)
                    return f"[SUCCESS] Working directory set to '{path}'. Git repository initialized."
                except Exception:
                    pass
        return f"[SUCCESS] Working directory set to '{path or '(none)'}'"

    def _projects_path(self, user_id: str) -> str:
        path = os.path.join(PROJECTS_DIR, f"{user_id}.json")
        os.makedirs(PROJECTS_DIR, exist_ok=True)
        return path

    def _load_projects(self, user_id: str) -> list:
        path = self._projects_path(user_id)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_projects(self, user_id: str, projects: list):
        path = self._projects_path(user_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(projects, f, indent=2)

    def get_recent_projects(self, user_id: str) -> list:
        return self._load_projects(user_id)

    def record_project_use(self, user_id: str, path: str):
        projects = self._load_projects(user_id)
        now = time.time()
        name = os.path.basename(os.path.normpath(path))
        pid = uuid.uuid5(uuid.NAMESPACE_DNS, path.lower()).hex
        projects = [p for p in projects if p.get("id") != pid]
        projects.insert(0, {"id": pid, "name": name, "path": path, "timestamp": now})
        self._save_projects(user_id, projects[:10])

    def remove_recent_project(self, user_id: str, project_id: str):
        projects = self._load_projects(user_id)
        projects = [p for p in projects if p.get("id") != project_id]
        self._save_projects(user_id, projects)

    def approve_tool(self, req_id: str, approved: bool):
        with self._approval_lock:
            entry = self.pending_approvals.get(req_id)
            if entry:
                entry["approved"] = approved
                entry["resolved"] = True
                entry.get("event", threading.Event()).set()

    def git_undo(self, user_id: str) -> str:
        state = self._get_user_state(user_id)
        work_dir = state.get("work_dir", "")
        if not work_dir or not os.path.isdir(work_dir):
            return "[ERROR] Working directory not set or does not exist"
        try:
            import subprocess
            save = subprocess.run(["git", "rev-parse", "HEAD"], cwd=work_dir, capture_output=True, text=True, timeout=10)
            if save.returncode == 0:
                state["undo_point"] = save.stdout.strip()
            result = subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=work_dir, capture_output=True, text=True, timeout=10)
            output = result.stdout[:3000] if result.stdout else ""
            if result.stderr:
                output += "\n" + result.stderr[:2000] if output else result.stderr[:2000]
            if "fatal" in (result.stdout + result.stderr).lower():
                return f"[ERROR] Undo failed: {output}"
            return f"[SUCCESS] Undo successful.\n{output}"
        except Exception as e:
            return f"[ERROR] git_undo: {e}"

    def git_redo(self, user_id: str) -> str:
        state = self._get_user_state(user_id)
        work_dir = state.get("work_dir", "")
        undo_point = state.get("undo_point")
        if not undo_point:
            return "[ERROR] Nothing to redo (no undo point saved)"
        if not work_dir or not os.path.isdir(work_dir):
            return "[ERROR] Working directory not set or does not exist"
        try:
            import subprocess
            result = subprocess.run(["git", "reset", "--hard", undo_point], cwd=work_dir, capture_output=True, text=True, timeout=10)
            state["undo_point"] = None
            output = result.stdout[:3000] if result.stdout else ""
            if result.stderr:
                output += "\n" + result.stderr[:2000] if output else result.stderr[:2000]
            if "fatal" in (result.stdout + result.stderr).lower():
                return f"[ERROR] Redo failed: {output}"
            return f"[SUCCESS] Redo successful.\n{output}"
        except Exception as e:
            return f"[ERROR] git_redo: {e}"

    def git_status(self, user_id: str) -> str:
        state = self._get_user_state(user_id)
        work_dir = state.get("work_dir", "")
        if not work_dir or not os.path.isdir(work_dir):
            return "[ERROR] Working directory not set or does not exist"
        try:
            import subprocess
            result = subprocess.run(["git", "status"], cwd=work_dir, capture_output=True, text=True, timeout=10)
            output = result.stdout[:3000] if result.stdout else ""
            if result.stderr:
                output += "\n" + result.stderr[:2000] if output else result.stderr[:2000]
            return output or "(no output)"
        except Exception as e:
            return f"[ERROR] git_status: {e}"

    def save_session(self, user_id: str, session_id: str = None, force: bool = False):
        state = self._get_user_state(user_id)
        sid = session_id or state["active_session"]
        if not sid or sid in ("null", "undefined"): return
        session_data = state["sessions"][sid]
        memory = session_data["messages"]
        if not memory and not force: return

        title = next((msg["content"][:30] + "..." for msg in memory if msg["role"] == "user"), "New Chat")
        user_dir = os.path.join(SESSIONS_DIR, user_id)
        os.makedirs(user_dir, exist_ok=True)
        filepath = os.path.join(user_dir, f"{sid}.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"id": sid, "title": title, "messages": memory, "attachments": session_data["attachments"], "timestamp": time.time()}, f)

    def load_session(self, user_id: str, session_id: str):
        state = self._get_user_state(user_id)
        user_dir = os.path.join(SESSIONS_DIR, user_id)
        filepath = os.path.join(user_dir, f"{session_id}.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                state["active_session"] = data["id"]
                state["sessions"][data["id"]] = {
                    "messages": data.get("messages", []),
                    "attachments": data.get("attachments", [])
                }
                return True
        return False

    def add_attachment(self, user_id: str, filename: str, session_id: str = None, display_name: str = None):
        state = self._get_user_state(user_id)
        sid = session_id or state["active_session"]
        if sid not in state["sessions"]:
            state["sessions"][sid] = {"messages": [], "attachments": []}
        attachment = {"filename": filename, "displayName": display_name or filename, "timestamp": time.time()}
        state["sessions"][sid]["attachments"].append(attachment)
        self.save_session(user_id, sid)
        return attachment

    def new_session(self, user_id: str, session_id: str = None):
        state = self._get_user_state(user_id)
        sid = session_id or uuid.uuid4().hex
        state["active_session"] = sid
        state["sessions"][sid] = {"messages": [], "attachments": []}

    def switch_model(self, user_id: str, model_name: str, session_id: str = None) -> str:
        state = self._get_user_state(user_id)
        state["model"] = model_name
        if session_id is None:
            for sid in state["sessions"]:
                state["sessions"][sid]["messages"] = []
        else:
            self._get_stm(user_id, session_id).clear()
        return f"[SWITCH] Switched to '{model_name}'"

    def clear_context(self, user_id: str, session_id: str = None):
        sid = session_id or self.get_active_session(user_id)
        self._get_stm(user_id, sid).clear()
        self.save_session(user_id, sid, force=True)
        return f"Context cleared for session {sid[:8]}..."

    def wipe_memory(self, user_id: str):
        self.db.clear(user_id=user_id)
        return f"All vector memory wiped for user {user_id}."

    def get_available_models(self, force_refresh: bool = False) -> list[str]:
        now = time.time()
        if not force_refresh and self._model_cache and (now - self._model_cache_time) < self._model_cache_ttl:
            return self._model_cache

        models = []
        try:
            resp = ollama.list()
            models_raw = resp.get("models", []) if isinstance(resp, dict) else resp.models
            models = [m.model if hasattr(m, "model") else m["model"] for m in models_raw]
        except Exception:
            pass
        cloud = self.provider_manager.list_cloud_models()
        for c in cloud:
            models.append(f"{c['provider']}:{c['model']}")

        self._model_cache = models
        self._model_cache_time = now
        return models

    def _parse_tool_calls(self, text: str) -> list[dict]:
        pattern = r'\[\[TOOL_CALL:\s*(\w+)\(([^)]*)\)\]\]'
        matches = re.findall(pattern, text)
        results = []
        for name, args_str in matches:
            args = {}
            if args_str.strip():
                arg_pattern = r'(\w+)=("(?:[^"\\]|\\.)*"|[^,)\s]+)'
                for m in re.finditer(arg_pattern, args_str):
                    key = m.group(1)
                    value = m.group(2).strip('"')
                    args[key] = value
            results.append({"name": name, "args": args})
        return results

    def _handle_tool_calls(self, response: str, state: dict, user_id: str, sid: str):
        """Post-process response for tool calls, execute them, and return follow-up context."""
        tool_calls = self._parse_tool_calls(response)
        if not tool_calls:
            return response, None

        personality = self.get_active_personality(user_id)
        work_dir = state.get("work_dir", "")
        tool_results = []

        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]

            if tool_name == "git_undo":
                result = self.git_undo(user_id)
                tool_results.append(f"Tool '{tool_name}' executed:\n{result}")
            elif tool_name == "git_redo":
                result = self.git_redo(user_id)
                tool_results.append(f"Tool '{tool_name}' executed:\n{result}")
            elif tool_name == "git_status":
                result = self.git_status(user_id)
                tool_results.append(f"Tool '{tool_name}' executed:\n{result}")
            elif tool_name in personality.requires_approval:
                req_id = uuid.uuid4().hex
                event = threading.Event()
                with self._approval_lock:
                    self.pending_approvals[req_id] = {
                        "resolved": False,
                        "approved": None,
                        "command": tool_args.get("command", ""),
                        "tool": tool_name,
                        "event": event,
                    }
                yield {"type": "approval", "req_id": req_id, "tool": tool_name, "args": tool_args}

                event.wait()

                with self._approval_lock:
                    approved = self.pending_approvals[req_id]["approved"]

                if approved:
                    result = personality.execute_tool(tool_name, tool_args, work_dir)
                    yield {"type": "approval_result", "approved": True, "result": result[:300]}
                    tool_results.append(f"Tool '{tool_name}' executed:\n{result}")
                else:
                    yield {"type": "approval_result", "approved": False}
                    tool_results.append(f"Tool '{tool_name}' was denied by the user.")
            else:
                try:
                    result = personality.execute_tool(tool_name, tool_args, work_dir)
                    tool_results.append(f"Tool '{tool_name}' executed:\n{result}")
                except ValueError as e:
                    tool_results.append(f"Error executing '{tool_name}': {e}")

        follow_up = "\n\n".join(tool_results)
        yield follow_up

    def _get_cancel_event(self, session_id: str) -> threading.Event:
        with self._cancel_lock:
            if session_id not in self._cancel_events:
                self._cancel_events[session_id] = threading.Event()
            return self._cancel_events[session_id]

    def cancel_stream(self, session_id: str):
        with self._cancel_lock:
            event = self._cancel_events.get(session_id)
            if event:
                event.set()

    def _check_cancelled(self, session_id: str) -> bool:
        with self._cancel_lock:
            event = self._cancel_events.get(session_id)
            return event is not None and event.is_set()

    def chat_stream(self, user_id: str, user_msg: str, session_id: str = None, attachments: str = None):
        user_msg = user_msg.strip()
        if not user_msg and not attachments:
            return

        if attachments:
            file_list = attachments.split(',')
            user_msg = f"[Attached files: {', '.join(file_list)}]\n{user_msg}"

        state = self._get_user_state(user_id)
        sid = session_id or state["active_session"]
        model = state["model"]
        stm = self._get_stm(user_id, sid)

        personality = self.get_active_personality(user_id)
        work_dir = state.get("work_dir", "")

        # Build context
        user_vec = _embed(user_msg) if True else None
        refers_to_file = any(t in user_msg.lower() for t in ["this file", "that file", "the pdf", "the document", "uploaded file", "inside that file"])
        is_general_query = any(t in user_msg.lower() for t in ["what is in", "summary", "overview", "everything", "list all"])

        recall_limit = 20 if is_general_query else 3
        past_context = self.db.recall(user_id=user_id, query_vector=user_vec, limit=recall_limit) if user_vec else ""

        if refers_to_file:
            session_context = self.db.recall_by_session(user_id, sid, limit=15)
            if session_context:
                past_context = f"{past_context}\n\n[Most Recent Documents]:\n{session_context}" if past_context else session_context

        should_search = any(t in user_msg.lower() for t in ["search", "who is", "city"]) or (not past_context.strip() and not is_general_query and not refers_to_file)
        if should_search:
            web_data = get_web_context(user_msg)
            if web_data:
                past_context = f"Web:\n{web_data}"
                self.db.remember(role="system_document", content=f"Learned from Web:\n{web_data}", user_id=user_id, session_id=sid)

        # Build personality-aware system prompt
        sys_template = personality.system_prompt
        sys_prompt = sys_template.replace("{context}", past_context).replace("{work_dir}", work_dir or "(not set)")
        active_context = [{"role": "system", "content": sys_prompt}] + stm + [{"role": "user", "content": user_msg}]

        try:
            provider = self.provider_manager.get_provider(model)
            full_response = ""
            tool_loop_count = 0
            was_cancelled = False

            while True:
                if tool_loop_count >= MAX_TOOL_CALL_LOOPS:
                    yield json.dumps({"token": "\n\n[SYSTEM] Max tool call depth reached. Stopping tool execution loop."})
                    break
                tool_loop_count += 1

                ans_chunks = []
                stream = provider.chat_stream(messages=active_context, stream=True)
                for chunk in stream:
                    if self._check_cancelled(sid):
                        was_cancelled = True
                        yield json.dumps({"token": "\n\n[Response cancelled by user]"})
                        break
                    ans_chunks.append(chunk)
                    yield json.dumps({"token": chunk})

                response_text = "".join(ans_chunks).strip()
                full_response += response_text

                # Check for tool calls
                gen = self._handle_tool_calls(response_text, state, user_id, sid)
                follow_up_text = None
                for event in gen:
                    if isinstance(event, dict):
                        # Yield approval/result events
                        yield json.dumps(event)
                        if event.get("type") == "approval_result":
                            if event.get("approved"):
                                follow_up_text = f"Tool executed successfully."
                            else:
                                follow_up_text = "Tool execution was denied."
                    else:
                        follow_up_text = event

                if follow_up_text is None:
                    break

                # Feed tool results back to LLM
                active_context.append({"role": "assistant", "content": response_text})
                active_context.append({"role": "system", "content": f"Tool result:\n{follow_up_text}\n\nNow respond to the user with the result."})
                # Loop continues for follow-up

            # Save session (skip if cancelled)
            if not was_cancelled:
                stm.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": full_response}])
                if len(stm) > STM_WINDOW:
                    del stm[:-STM_WINDOW]
                self.save_session(user_id, sid)
                self.db.remember("user", user_msg, user_id=user_id, session_id=sid, precomputed_vector=user_vec)
                self.db.remember("assistant", full_response, user_id=user_id, session_id=sid)

        except ProviderError as e:
            if e.code == CONTEXT_LENGTH_EXCEEDED and len(stm) > 2:
                yield json.dumps({"token": f"\n\n[SYSTEM] Context too long. Trimming oldest messages and retrying..."})
                del stm[:max(1, len(stm) - 2)]
                self.save_session(user_id, sid)
                yield json.dumps({"token": f"\n[SYSTEM] Please resend your message. Context was trimmed to fit."})
            else:
                yield json.dumps({"token": f"\n[{e.code}] {sanitize_error(str(e))}"})
        except Exception as e:
            yield json.dumps({"token": f"[ERROR] {sanitize_error(str(e))}"})
        finally:
            with self._cancel_lock:
                self._cancel_events.pop(sid, None)

    def read_legal_pdf(self, path: str, user_id: str, session_id: str):
        return ingest_pdf(path, self.db, user_id, session_id)

    def draw(self, prompt: str, seed=None):
        return self.img_service.draw(prompt, seed)

    def _shutdown(self):
        self.img_service.unload()
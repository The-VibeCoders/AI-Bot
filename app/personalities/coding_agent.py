import os
import subprocess
import json
import shlex
import time
from .base import BasePersonality


class CodingAgentPersonality(BasePersonality):
    @property
    def id(self) -> str:
        return "coding_agent"

    @property
    def name(self) -> str:
        return "Coding Agent"

    @property
    def description(self) -> str:
        return "Autonomous coding assistant with file editing, shell commands, and git version control"

    @property
    def icon(self) -> str:
        return "⌨️"

    @property
    def system_prompt(self) -> str:
        return (
            "You are Doremon in Coding Agent mode. You are an autonomous software engineering assistant "
            "running on the user's machine. You can read, write, and patch files, list directory contents, "
            "and execute shell commands. You also have access to git for version control.\n\n"
            "When you need to interact with files or the system, use the available tools by outputting "
            "a tool call in this exact format:\n"
            "[[TOOL_CALL: tool_name(arg1=\"value1\", arg2=\"value2\")]]\n\n"
            "Available tools:\n"
            "- list_directory(path=\"...\"): List files and folders in a directory\n"
            "- read_file(filepath=\"...\"): Read the contents of a file\n"
            "- write_file(filepath=\"...\", content=\"...\"): Write content to a file (creates/overwrites)\n"
            "- patch_file(filepath=\"...\", find=\"...\", replace=\"...\"): Replace exact text in a file\n"
            "- execute_shell(command=\"...\"): Run a shell command (requires user approval)\n"
            "- git_status(): Show current git status\n"
            "- git_commit(message=\"...\"): Stage all changes and commit\n"
            "- git_undo(): Undo last commit (reset --hard HEAD~1)\n"
            "- git_redo(): Redo a previously undone commit\n"
            "- get_working_directory(): Show the current working directory\n\n"
            "Rules:\n"
            "1. You can ONLY modify files inside the working directory.\n"
            "2. For execute_shell, explain what the command does before calling it.\n"
            "3. After any file edit, suggest running relevant tests or verification.\n"
            "4. Use git_status to check the state of the repository before making changes.\n\n"
            "Current working directory: {work_dir}\n"
            "=== LIVE CONTEXT ===\n{context}\n==================="
        )

    @property
    def tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files and directories in a given path",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path to list"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the full contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {"type": "string", "description": "Path to the file"}
                        },
                        "required": ["filepath"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file (creates new file or overwrites existing)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {"type": "string", "description": "Path to the file"},
                            "content": {"type": "string", "description": "Full file content to write"}
                        },
                        "required": ["filepath", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "patch_file",
                    "description": "Find and replace exact text in an existing file. Use for surgical edits.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {"type": "string", "description": "Path to the file"},
                            "find": {"type": "string", "description": "Exact text to find"},
                            "replace": {"type": "string", "description": "Replacement text"}
                        },
                        "required": ["filepath", "find", "replace"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_shell",
                    "description": "Execute a shell command. Requires user approval.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Shell command to execute"}
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "git_status",
                    "description": "Show the working tree status (modified, staged, untracked files)",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "git_commit",
                    "description": "Stage all changes and commit with a descriptive message",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "Commit message"}
                        },
                        "required": ["message"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "git_undo",
                    "description": "Undo the last commit (git reset --hard HEAD~1). WARNING: discards changes.",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "git_redo",
                    "description": "Redo a previously undone commit (restore from saved undo point).",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_working_directory",
                    "description": "Show the currently set working directory",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]

    @property
    def requires_approval(self) -> list[str]:
        return ["execute_shell"]

    def execute_tool(self, tool_name: str, arguments: dict, work_dir: str = "") -> str:
        if tool_name == "list_directory":
            path = arguments.get("path", ".")
            if work_dir:
                path = os.path.join(work_dir, path) if not os.path.isabs(path) else path
            return self._list_directory(path)
        elif tool_name == "read_file":
            filepath = arguments.get("filepath", "")
            if work_dir:
                filepath = os.path.join(work_dir, filepath) if not os.path.isabs(filepath) else filepath
            return self._read_file(filepath)
        elif tool_name == "write_file":
            filepath = arguments.get("filepath", "")
            content = arguments.get("content", "")
            if work_dir:
                filepath = os.path.join(work_dir, filepath) if not os.path.isabs(filepath) else filepath
            return self._write_file(filepath, content)
        elif tool_name == "patch_file":
            filepath = arguments.get("filepath", "")
            find = arguments.get("find", "")
            replace = arguments.get("replace", "")
            if work_dir:
                filepath = os.path.join(work_dir, filepath) if not os.path.isabs(filepath) else filepath
            return self._patch_file(filepath, find, replace)
        elif tool_name == "execute_shell":
            return self._execute_shell(arguments.get("command", ""))
        elif tool_name == "git_status":
            return self._git_command(["status"], work_dir)
        elif tool_name == "git_commit":
            msg = arguments.get("message", "Auto-commit")
            self._git_command(["add", "."], work_dir)
            return self._git_command(["commit", "-m", msg], work_dir)
        elif tool_name == "git_undo":
            return "Use the agent-level git_undo to track undo history."
        elif tool_name == "git_redo":
            return "Use the agent-level git_redo to restore from undo point."
        elif tool_name == "get_working_directory":
            return f"Current working directory: {work_dir or 'Not set'}"
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def get_sidebar_ui(self, user_state: dict) -> str:
        work_dir = user_state.get("work_dir", "")
        return f"""
<div class="personality-section">
  <div class="section-label">Working Directory</div>
  <div class="work-dir-input">
    <input type="text" id="work-dir-input" value="{work_dir}" placeholder="C:\\Path\\To\\Project" style="width:100%;padding:8px;border-radius:6px;background:#1e1e1e;color:#fff;border:1px solid #333;font-size:12px;box-sizing:border-box;" />
    <button id="set-work-dir-btn" class="tool-btn" style="justify-content:center;margin-top:6px;padding:6px;">Set Directory</button>
    <div id="work-dir-status" style="font-size:11px;color:#888;margin-top:4px;"></div>
  </div>
  <div style="display:flex;gap:6px;margin-top:8px;">
    <button id="git-undo-btn" class="tool-btn" style="flex:1;justify-content:center;padding:6px;font-size:12px;" title="Undo last commit">↩ Undo</button>
    <button id="git-redo-btn" class="tool-btn" style="flex:1;justify-content:center;padding:6px;font-size:12px;" title="Redo undone commit">↪ Redo</button>
  </div>
</div>"""

    def _list_directory(self, path: str) -> str:
        try:
            if not os.path.exists(path):
                return f"[ERROR] Path does not exist: {path}"
            items = os.listdir(path)
            result = []
            for item in sorted(items):
                full = os.path.join(path, item)
                if os.path.isdir(full):
                    result.append(f"📁 {item}/")
                elif os.path.isfile(full):
                    size = os.path.getsize(full)
                    result.append(f"📄 {item} ({size} bytes)")
                else:
                    result.append(f"❓ {item}")
            return "\n".join(result) if result else "(empty directory)"
        except Exception as e:
            return f"[ERROR] list_directory: {e}"

    def _read_file(self, filepath: str) -> str:
        try:
            if not os.path.exists(filepath):
                return f"[ERROR] File not found: {filepath}"
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if len(content) > 100000:
                content = content[:100000] + "\n\n[TRUNCATED: file too large, showing first 100000 chars]"
            return content
        except Exception as e:
            return f"[ERROR] read_file: {e}"

    def _write_file(self, filepath: str, content: str) -> str:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            size = len(content)
            return f"[SUCCESS] Written {size} bytes to {os.path.basename(filepath)}"
        except Exception as e:
            return f"[ERROR] write_file: {e}"

    def _patch_file(self, filepath: str, find: str, replace: str) -> str:
        try:
            if not os.path.exists(filepath):
                return f"[ERROR] File not found: {filepath}"
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if find not in content:
                return f"[ERROR] Could not find the specified text in {os.path.basename(filepath)}. The text must match exactly."
            new_content = content.replace(find, replace, 1)
            count = content.count(find)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
            return f"[SUCCESS] Replaced 1 occurrence in {os.path.basename(filepath)} ({count} total matches found)"
        except Exception as e:
            return f"[ERROR] patch_file: {e}"

    def _execute_shell(self, command: str) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )
            output = ""
            if result.stdout:
                output += f"stdout:\n{result.stdout[:5000]}"
            if result.stderr:
                if output:
                    output += "\n"
                output += f"stderr:\n{result.stderr[:5000]}"
            if result.returncode != 0:
                output += f"\n[Exit code: {result.returncode}]"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "[ERROR] Command timed out after 60 seconds"
        except Exception as e:
            return f"[ERROR] execute_shell: {e}"

    def _git_command(self, args: list[str], work_dir: str) -> str:
        try:
            if not work_dir or not os.path.isdir(work_dir):
                return "[ERROR] Working directory not set or does not exist"
            result = subprocess.run(
                ["git"] + args,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            output = ""
            if result.stdout:
                output += result.stdout[:3000]
            if result.stderr:
                if output:
                    output += "\n"
                output += result.stderr[:2000]
            if result.returncode != 0 and not output:
                output = f"git returned exit code {result.returncode}"
            return output or "(no output)"
        except FileNotFoundError:
            return "[ERROR] Git is not installed or not in PATH"
        except Exception as e:
            return f"[ERROR] git: {e}"

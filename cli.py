from app.core.agent import LocalDoremonMaster
from app.utils.compat import _extract_chunk_token
import uuid
import sys

def print_guide():
    print("\n" + "="*55)
    print(" 🤖 DOREMON LOCAL AI - CLI GUIDE")
    print("="*55)
    print(" Commands:")
    print("  /draw <prompt>    - Generate an image")
    print("  /read <path>      - Ingest a PDF into memory")
    print("  /model <name>     - Switch LLM model")
    print("  /list             - List all available models")
    print("  /clear            - Clear current chat context")
    print("  /wipe             - Wipe all vector memory")
    print("  /help             - Show this guide")
    print("  /quit             - Exit the program")
    print("-" * 55)
    print(" Tip: Just type normally to chat with the AI!")
    print("="*55 + "\n")

def run_cli():
    bot = LocalDoremonMaster()
    print("-" * 55)
    print("Doremon Local AI Modular CLI")
    print("-" * 55)
    
    user_id = input("Enter username to login to CLI: ").strip() or "cli_user"
    bot._init_user(user_id)
    session_id = bot.get_active_session(user_id)
    print(f"Logged in as '{user_id}'. Session: {session_id[:8]}...")
    
    print_guide()

    while True:
        try:
            ui = input("\nYou: ").strip()
            if not ui: continue
            
            # Command handling
            if ui.lower() in ("q", "exit", "quit"): 
                print("Goodbye! 👋")
                break
            elif ui.lower() == "/help":
                print_guide()
                continue
            elif ui.startswith("/draw "):
                msg, _ = bot.draw(ui[6:])
                print(f"🎨 {msg}")
            elif ui.startswith("/read "):
                print(f"📄 {bot.read_legal_pdf(ui[6:], user_id=user_id, session_id=session_id)}")
            elif ui.startswith("/model "):
                model_name = ui[7:].strip()
                if model_name:
                    print(bot.switch_model(user_id, model_name))
                else:
                    print("Error: Please provide a model name. Example: /model llama3.2")
            elif ui.lower() == "/list":
                models = bot.get_available_models()
                print(f"Available models: {', '.join(models) if models else 'None found'}")
                print(f"Active model: {bot.get_current_model(user_id)}")
            elif ui.lower() == "/clear":
                print(bot.clear_context(user_id, session_id))
            elif ui.lower() == "/wipe":
                confirm = input("Are you sure you want to wipe ALL memory? (y/n): ")
                if confirm.lower() == 'y':
                    print(bot.wipe_memory(user_id))
                else:
                    print("Wipe cancelled.")
            else:
                # Regular chat
                for token in bot.chat_stream(user_id=user_id, user_msg=ui, session_id=session_id):
                    print(token, end="", flush=True)
                print()
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    run_cli()

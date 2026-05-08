from app.core.agent import LocalDoremonMaster
from app.utils.compat import _extract_chunk_token

def run_cli():
    bot = LocalDoremonMaster()
    print("-" * 55 + "\nDoremon Local AI Modular CLI\n" + "-" * 55)

    while True:
        try:
            ui = input("\nYou: ").strip()
            if not ui: continue
            if ui.lower() in ("q", "exit", "quit"): break
            
            if ui.startswith("/draw "):
                msg, _ = bot.draw(ui[6:])
                print(msg)
            elif ui.startswith("/read "):
                print(bot.read_legal_pdf(ui[6:]))
            else:
                for token in bot.chat_stream(ui):
                    print(token, end="", flush=True)
                print()
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    run_cli()
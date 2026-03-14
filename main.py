import streamlit as st
import os
import time
from bot import LocalDoremonMaster  # This imports your entire backend engine!
import json
# ==========================================
# 🎨 STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Doremon Local AI", page_icon="🤖", layout="wide")
st.title("🤖 Agent: Privacy-First AI Agent")
st.caption("Powered by Llama 3.2 3B & Realistic Vision | RTX 3050 Ti Optimized")

# ==========================================
# 🧠 SESSION STATE MANAGEMENT
# ==========================================
# Streamlit re-runs the whole script on every click. 
# We use session_state to keep your bot and chat history alive in RAM.
if "agent" not in st.session_state:
    with st.spinner("Booting up Local AI Engine..."):
        st.session_state.agent = LocalDoremonMaster()
        
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 🎛️ THE SIDEBAR (Controls & Uploads)
# ==========================================
with st.sidebar:
    st.title("Doremon Agent")
    
    # Create clean, modern tabs in the sidebar
    tab_controls, tab_history = st.tabs(["⚙️ Controls", "📜 DB History"])
    
    # --- TAB 1: CONTROLS & UPLOADS ---
    with tab_controls:
        st.header("Agent Settings")
        
        # 1. Dynamic Model Switcher UI
        available_models = st.session_state.agent.get_available_models()
        if available_models:
            selected_model = st.selectbox(
                "Active LLM", 
                available_models, 
                index=available_models.index(st.session_state.agent.current_model) if st.session_state.agent.current_model in available_models else 0
            )
            if selected_model != st.session_state.agent.current_model:
                st.session_state.agent.switch_model(selected_model)
                st.success(f"Switched to {selected_model}")
                
        st.divider()
        
        # 2. PDF Upload UI
        st.subheader("📄 RAG Document Ingestion")
        uploaded_file = st.file_uploader("Upload Your files Here(Upload pdfs only)", type="pdf")
        if uploaded_file is not None:
            if st.button("Vectorize Document"):
                with st.spinner("Chunking and saving to Long-Term Memory..."):
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    result = st.session_state.agent.read_legal_pdf(temp_path)
                    st.success(result)
                    os.remove(temp_path)

        st.divider()
        st.markdown("**Commands:**\n- Type normally to chat\n- `/draw [prompt]` for art\n- `/explain [file]` for code")

    # --- TAB 2: LONG-TERM MEMORY VIEWER ---
    with tab_history:
        st.subheader("Vector DB Logs")
        st.caption("Permanent records saved to doremon_memory.jsonl")
        
        if st.button("🔄 Refresh DB Logs"):
            if os.path.exists("doremon_memory.jsonl"):
                with open("doremon_memory.jsonl", "r", encoding="utf-8") as f:
                    history_logs = [json.loads(line) for line in f]
                
                # Filter out the massive PDF chunks to keep the UI clean
                chat_logs = [log for log in history_logs if log['role'] in ['user', 'assistant']]
                
                # Show the last 10 interactions, newest at the top
                for log in reversed(chat_logs[-20:]):
                    with st.chat_message(log["role"]):
                        st.markdown(log["content"])
            else:
                st.info("No long-term memory found yet.")
                
        st.divider()
        
        # Bonus: A kill-switch to wipe the database for fresh presentations
        if st.button("🗑️ Wipe Database", type="primary"):
            if os.path.exists("doremon_memory.jsonl"):
                os.remove("doremon_memory.jsonl")
            
            # Re-initialize the backend DB and clear the frontend screen
            st.session_state.agent.db.__init__() 
            st.session_state.messages = []
            st.rerun() # Forces the web app to refresh instantly

# ==========================================
# 💬 THE CHAT INTERFACE
# ==========================================
# Display all past messages in the UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "image" in msg:
            st.image(msg["image"]) # Displays generated images in the chat stream!

# Capture user input
if prompt := st.chat_input("Ask Doremon or type /draw..."):
    # 1. Show user message in UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Process the command
    with st.chat_message("assistant"):
        if prompt.startswith('/draw '):
            with st.spinner("🎨 Allocating GPU VRAM for Art Generation..."):
                response = st.session_state.agent.draw(prompt[6:])
                st.markdown(response)
                
                # If an image was successfully created, find the filename and display it
                if "Saved as" in response or "saved as" in response:
                    filename = response.split("'")[1] # Extracts the filename from your backend's string
                    if os.path.exists(filename):
                        st.image(filename)
                        # Save image info to session state so it doesn't disappear on scroll
                        st.session_state.messages.append({"role": "assistant", "content": response, "image": filename})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
        else:
            with st.spinner("Thinking..."):
                response = st.session_state.agent.chat(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
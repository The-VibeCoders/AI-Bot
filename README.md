Personal AI Chatbot / Influencer Model 🤖

An interactive, locally-hosted AI chatbot built with Python, Streamlit, and Ollama. This project demonstrates the integration of modern generative AI tools with a web-based frontend to create customizable conversational agents and persona models.

🚀 Overview

This application leverages Ollama to run Large Language Models (LLMs) entirely locally, ensuring data privacy and reducing API costs. The frontend is powered by Streamlit, providing a clean, responsive, and interactive chat interface. By utilizing custom system prompts, the chatbot can be tailored to adopt specific personas, acting as a foundation for "AI influencer" modeling.

🛠️ Tech Stack

Language: Python 3.x

Frontend UI: Streamlit

AI/LLM Engine: Ollama (Local Inference)

Models: Llama 3 / Mistral (or specify your preferred local model)

✨ Key Features

Local AI Inference: Runs entirely on your local machine using Ollama, requiring no external API keys (like OpenAI) and ensuring complete data privacy.

Interactive Chat Interface: A sleek, WhatsApp-style messaging UI built rapidly with Streamlit's chat elements.

Persona Customization: Easily swap out system prompts to change the AI's behavior, tone, and knowledge base (e.g., from a helpful coding assistant to a specific influencer persona).

Session State Management: Retains conversation history within the active session for context-aware responses.

⚙️ Installation & Setup

Follow these steps to get the project running on your local machine:

1. Install Ollama

Download and install Ollama from ollama.com.
Once installed, open your terminal and pull a model (e.g., Llama 3):

ollama run llama3


2. Clone the Repository

git clone [[https://github.com/Kaustubh-465/your-repo-name.git](https://github.com/Kaustubh-465/your-repo-name.git)](https://github.com/The-VibeCoders/AI-Bot)
cd your-repo-name


3. Set Up a Virtual Environment (Recommended)

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate


4. Install Dependencies

pip install streamlit
# Add any other required libraries here, e.g., requests, langchain, etc.


💻 Usage

Start the Streamlit development server:

streamlit run app.py


The application will automatically open in your default web browser (usually at http://localhost:8501).

🔮 Future Enhancements

[ ] Add Retrieval-Augmented Generation (RAG) to allow the bot to answer questions based on custom PDF documents.

[ ] Implement a database (SQLite/MySQL) to save chat histories across different sessions.

[ ] Add Voice-to-Text capabilities for hands-free interaction.

[ ] Containerize the application using Docker for easier deployment.

Developed by The Vibecoders

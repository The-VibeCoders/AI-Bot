import os

# Root directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")

MAX_HOT_MEMORIES = 500
STM_WINDOW = 6
EMBED_MODEL = "nomic-embed-text"
SD_MODEL_ID = "stablediffusionapi/realistic-vision-v51"
VECTOR_DIM = 768
MAX_TOOL_CALL_LOOPS = 10

LOCAL_USER = "local_user"
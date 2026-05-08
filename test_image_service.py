import sys
sys.path.append()
from app.services.image_service import ImageService

service = ImageService()
print("Service initialized. Device:", service.device)
prompt = "A simple red circle"
message, filename = service.draw(prompt)
print("Message:", message)
print("Filename:", filename)

import sys
sys.path.append("C:\\Users\\divek\\Desktop\\mini project v2")
from app.services.image_service import ImageService

service = ImageService()
print("Service initialized. Device:", service.device)
prompt = "A simple red circle"
message, filename = service.draw(prompt)
print("Message:", message)
print("Filename:", filename)

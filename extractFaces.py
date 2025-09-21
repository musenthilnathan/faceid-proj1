
from deepface import DeepFace
from PIL import Image
import os
import subprocess


# Ask user for group photo path
img_path = input("Enter path to group photo: ").strip()

# If HEIC, convert to JPG
if img_path.lower().endswith('.heic'):
	jpg_path = os.path.splitext(img_path)[0] + ".jpg"
	print(f"Converting {img_path} to {jpg_path}...")
	# Use ImageMagick's 'magick' command for conversion
	result = subprocess.run(["magick", "convert", img_path, jpg_path])
	if result.returncode != 0:
		raise RuntimeError(f"Failed to convert {img_path} to JPG. Make sure ImageMagick is installed and supports HEIC.")
	img_path = jpg_path

# Detect faces
faces = DeepFace.extract_faces(img_path=img_path, detector_backend="retinaface")

print(f"Found {len(faces)} face(s)")


# Loop over all detected faces
img = Image.open(img_path)
for idx, face in enumerate(faces):
	region = face["facial_area"]
	x, y, w, h = region["x"], region["y"], region["w"], region["h"]
	cropped = img.crop((x, y, x+w, y+h))
	cropped.show()
	filename = input(f"Enter filename for face {idx+1} (without extension): ")
	save_path = f"{filename}.jpg"
	cropped.save(save_path)
	print(f"Cropped face saved to {save_path}")

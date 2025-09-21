import cv2
import os
import subprocess
import numpy as np
import sqlite3
import tkinter as tk
from tkinter import filedialog
from deepface import DeepFace
import pickle

DB = "faces.db"

def erase_db():
    if os.path.exists(DB):
        os.remove(DB)
        print(f"Database {DB} erased.")
    else:
        print(f"Database {DB} does not exist.")

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS faces(
                 person TEXT, embedding BLOB)""")
    conn.commit()
    conn.close()

def store_embeddings(person, files):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    for f in files:
        try:
            emb = DeepFace.represent(img_path=f, model_name="Facenet")[0]["embedding"]
            # L2 normalize embedding
            emb = np.array(emb)
            emb = emb / np.linalg.norm(emb)
            print(f"[DEBUG] Storing embedding for {person} from {f}: type={type(emb)}, len={len(emb)}, sample={emb[:5]}")
            c.execute("INSERT INTO faces VALUES(?, ?)", (person, pickle.dumps(emb.tolist())))
        except Exception as e:
            print("Skip", f, ":", e)
    conn.commit()
    conn.close()

def capture_from_camera(save_path):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return None
    print("\n=== Camera Window Instructions ===")
    print("1. Look for a window titled 'Camera Capture'")
    print("2. If you don't see it, check your taskbar or behind other windows")
    print("3. Click inside the image window to capture")
    print("4. Press ESC to cancel")
    print("==============================\n")
    
    captured = [False]
    frame_holder = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            captured[0] = True
    
    cv2.namedWindow("Camera Capture", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Camera Capture", 100, 100)
    cv2.setMouseCallback("Camera Capture", on_mouse)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        frame_holder[0] = frame
        cv2.imshow("Camera Capture", frame)
        
        # Try to bring window to front periodically
        try:
            cv2.setWindowProperty("Camera Capture", cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        
        k = cv2.waitKey(1)
        if k%256 == 27:  # ESC pressed
            print("Capture cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return None
        if captured[0]:
            cv2.imwrite(save_path, frame_holder[0])
            print(f"Image saved to {save_path}")
            cap.release()
            cv2.destroyAllWindows()
            return save_path

def identify(file):
    try:
        test_emb = DeepFace.represent(img_path=file, model_name="Facenet")[0]["embedding"]
        # L2 normalize test embedding
        test_emb = np.array(test_emb)
        test_emb = test_emb / np.linalg.norm(test_emb)
        print(f"[DEBUG] Test embedding: type={type(test_emb)}, len={len(test_emb)}, sample={test_emb[:5]}")
    except Exception as e:
        print("No face found:", e)
        return

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT person, embedding FROM faces")
    rows = c.fetchall()
    conn.close()

    # Calculate distances for all known faces
    results = []
    for person, emb in rows:
        known_emb = np.array(pickle.loads(emb))
        print(f"[DEBUG] Comparing to {person}: type={type(known_emb)}, len={len(known_emb)}, sample={known_emb[:5]}")
        dist = np.linalg.norm(test_emb - known_emb)
        results.append((person, dist))

    if not results:
        print("No known faces in database.")
        return

    # Print all matches with confidence
    print("Match results:")
    threshold = 1.0
    best_idx = np.argmin([d for _, d in results])
    best_person, best_dist = results[best_idx]
    for person, dist in results:
        print(f"  {person}: distance={dist:.4f}")

    print()
    if best_dist < threshold:
        print(f"Match: {best_person} (distance={best_dist:.4f})")
    else:
        print("No match found (all distances above threshold)")

def process_group_photo(file_path):
    print(f"Processing group photo: {file_path}")
    if not file_path:
        print("No file selected")
        return
        
    # Handle HEIC format
    if file_path.lower().endswith('.heic'):
        jpg_path = os.path.splitext(file_path)[0] + ".jpg"
        print(f"Converting {file_path} to {jpg_path}...")
        result = subprocess.run(["magick", "convert", file_path, jpg_path])
        if result.returncode != 0:
            print(f"Failed to convert {file_path} to JPG. Make sure ImageMagick is installed and supports HEIC.")
            return
        file_path = jpg_path

    print("\nProcessing faces... This may take a moment.")
    # Extract faces using DeepFace
    faces = DeepFace.extract_faces(img_path=file_path, detector_backend="retinaface")
    if not faces:
        print("No faces found in the group photo.")
        return

    # Load the image and show thumbnails
    img = cv2.imread(file_path)
    print(f"\nFound {len(faces)} faces in the group photo.")
    print("\n=== Face Grid Window Instructions ===")
    print("1. A window titled 'Group Photo - Face Grid' will appear")
    print("2. If you don't see it, check your taskbar or behind other windows")
    print("3. Each face is numbered in green (top-left corner)")
    print("4. You'll be prompted to enter names for each face")
    print("5. Press any key to continue when ready")
    print("=====================================\n")
    
    # Process and arrange faces in a grid
    face_images = []
    max_height = 0
    max_width = 0
    
    # First pass: extract and process all faces
    for idx, face in enumerate(faces, 1):
        region = face["facial_area"]
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        
        # Add padding around face
        pad = 20
        y_start = max(0, y - pad)
        y_end = min(img.shape[0], y + h + pad)
        x_start = max(0, x - pad)
        x_end = min(img.shape[1], x + w + pad)
        
        face_img = img[y_start:y_end, x_start:x_end].copy()
        
        # Enhance image
        face_img_float = face_img.astype(float)
        if face_img_float.max() > 0:
            face_img_float = (face_img_float - face_img_float.min()) * 255 / (face_img_float.max() - face_img_float.min())
        face_img = face_img_float.astype(np.uint8)
        
        # Add white border
        face_img = cv2.copyMakeBorder(face_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        # Adjust contrast and brightness
        face_img = cv2.convertScaleAbs(face_img, alpha=1.2, beta=10)
        
        # Add number to corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(face_img, str(idx), (10, 30), font, 1, (0, 255, 0), 2)
        
        face_images.append(face_img)
        max_height = max(max_height, face_img.shape[0])
        max_width = max(max_width, face_img.shape[1])
    
    # Calculate grid layout
    n_faces = len(face_images)
    grid_size = int(np.ceil(np.sqrt(n_faces)))
    
    # Create blank canvas for grid
    canvas = np.ones((max_height * ((n_faces-1)//grid_size + 1), 
                     max_width * grid_size, 3), dtype=np.uint8) * 255
    
    # Place faces in grid
    for idx, face_img in enumerate(face_images):
        i, j = idx // grid_size, idx % grid_size
        
        # Center the face in its grid cell
        y_offset = i * max_height + (max_height - face_img.shape[0]) // 2
        x_offset = j * max_width + (max_width - face_img.shape[1]) // 2
        
        canvas[y_offset:y_offset + face_img.shape[0],
               x_offset:x_offset + face_img.shape[1]] = face_img
    
    # Show grid
    cv2.namedWindow("Group Photo - Face Grid", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Group Photo - Face Grid", 100, 100)
    cv2.imshow("Group Photo - Face Grid", canvas)
    cv2.setWindowProperty("Group Photo - Face Grid", cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(0)  # Wait for key press
    
    # Save grid for reference
    cv2.imwrite("all_faces_grid.jpg", canvas)
    print("\nSaved all faces to all_faces_grid.jpg")
    
    # Get names for each face while showing the grid
    print("\nLooking at the grid image with numbered faces:")
    cv2.setWindowProperty("Group Photo - Face Grid", cv2.WND_PROP_TOPMOST, 1)
    for idx, face_img in enumerate(face_images, 1):
        name = input(f"\nEnter name for face #{idx} (or press Enter to skip): ").strip()
        if name:
            # Save individual face
            save_path = f"{name}_group_{idx}.jpg"
            cv2.imwrite(save_path, face_img)
            # Store in database
            store_embeddings(name, [save_path])
            print(f"Stored face {idx} as {name}")
            # Refresh window to keep it visible
            cv2.setWindowProperty("Group Photo - Face Grid", cv2.WND_PROP_TOPMOST, 1)
    
    # Clean up
    cv2.destroyAllWindows()
    print("\nFinished processing group photo!")

def choose_files(multiple=True):
    try:
        root = tk.Tk()
        root.withdraw()
        if multiple:
            result = filedialog.askopenfilenames(title="Select images")
        else:
            result = filedialog.askopenfilename(title="Pick test image")
        root.destroy()
        return result
    except Exception as e:
        print(f"Error in file dialog: {e}")
        return None

def get_group_photo():
    method = input("Get group photo from (f)ile, (c)amera, or (q)uit? [f/c/q]: ").strip().lower()
    print(f"Selected method: {method}")
    if method == 'q':
        return None
    elif method == 'c':
        save_path = "group_photo.jpg"
        return capture_from_camera(save_path)
    else:
        print("Opening file dialog...")
        file_path = choose_files(multiple=False)
        print(f"Selected file: {file_path}")
        return file_path

if __name__ == "__main__":
    erase = input("Erase the database and start fresh? (y/n): ").strip().lower()
    if erase == 'y':
        erase_db()
    init_db()

    print("=== Face Recognition System ===")
    while True:
        print("\nOptions:")
        print("1. Tag faces from group photo")
        print("2. Add individual faces")
        print("3. Test face recognition")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            photo_path = get_group_photo()
            if photo_path:
                process_group_photo(photo_path)
        elif choice == '2':
            print("\n--- Adding Individual Faces ---")
            # Add individual faces menu
            while True:
                # Show known people
                conn = sqlite3.connect(DB)
                c = conn.cursor()
                c.execute("SELECT person, COUNT(*) FROM faces GROUP BY person")
                known_people = c.fetchall()
                conn.close()
                if known_people:
                    print("\nKnown people in the database:")
                    for idx, (pname, pcount) in enumerate(known_people, 1):
                        print(f"  {idx}. {pname}  |  No of images available: {pcount}")
                else:
                    print("\nNo known people in the database yet.")
                
                person_input = input("\nEnter number/name to add images for (or press Enter to go back): ").strip()
                if not person_input:
                    break
                
                # Resolve name
                name = None
                if person_input.isdigit() and known_people:
                    idx = int(person_input) - 1
                    if 0 <= idx < len(known_people):
                        name = known_people[idx][0]
                if not name:
                    name = person_input
                
                # Get images
                method = input("Add images from (f)ile or (c)amera? [f/c]: ").strip().lower()
                if method == 'c':
                    save_path = f"{name}_cam.jpg"
                    img_path = capture_from_camera(save_path)
                    if img_path:
                        store_embeddings(name, [img_path])
                else:
                    files = choose_files(multiple=True)
                    if files:
                        store_embeddings(name, files)
        
        elif choice == '3':
            # Test face recognition
            while True:
                print("\n--- Face Recognition Test ---")
                method = input("Test image from (f)ile, (c)amera, or (b)ack? [f/c/b]: ").strip().lower()
                if method == 'b':
                    break
                elif method == 'c':
                    test_path = "test_cam.jpg"
                    img_path = capture_from_camera(test_path)
                    if img_path:
                        identify(img_path)
                else:
                    test = choose_files(multiple=False)
                    if test:
                        identify(test)
        
        elif choice == '4':
            print("\nExiting program. Goodbye!")
            break

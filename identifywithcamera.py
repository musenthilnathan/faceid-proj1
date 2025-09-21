import cv2
import os
import subprocess

def erase_db():
    if os.path.exists(DB):
        os.remove(DB)
        print(f"Database {DB} erased.")
    else:
        print(f"Database {DB} does not exist.")
import sqlite3, tkinter as tk
from tkinter import filedialog
from deepface import DeepFace
import numpy as np, pickle

DB = "faces.db"

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
    print("Click inside the image to capture, or press ESC to cancel.")
    print("If you do not see the camera window, check behind your VS Code window or on your taskbar.")
    captured = [False]
    frame_holder = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            captured[0] = True
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    # Try to bring the window to the front (works on most platforms)
    try:
        cv2.setWindowProperty("Camera", cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    cv2.setMouseCallback("Camera", on_mouse)
    import subprocess, time
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        frame_holder[0] = frame
        cv2.imshow("Camera", frame)
        # Move window to (100, 100) and try to keep it on top
        cv2.moveWindow("Camera", 100, 100)
        try:
            cv2.setWindowProperty("Camera", cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        # Add a short delay before using wmctrl to allow the window to appear
        time.sleep(0.1)
        try:
            subprocess.run(['wmctrl', '-r', 'Camera', '-b', 'add,above'], check=False)
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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return None
    print("Press SPACE to capture, ESC to cancel.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow("Camera", frame)
        k = cv2.waitKey(1)
        if k%256 == 27:  # ESC pressed
            print("Capture cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return None
        elif k%256 == 32:  # SPACE pressed
            cv2.imwrite(save_path, frame)
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
    # Handle HEIC format
    if file_path.lower().endswith('.heic'):
        jpg_path = os.path.splitext(file_path)[0] + ".jpg"
        print(f"Converting {file_path} to {jpg_path}...")
        # Use ImageMagick's 'magick' command for conversion
        result = subprocess.run(["magick", "convert", file_path, jpg_path])
        if result.returncode != 0:
            print(f"Failed to convert {file_path} to JPG. Make sure ImageMagick is installed and supports HEIC.")
            return
        file_path = jpg_path

    # Extract faces using DeepFace
    faces = DeepFace.extract_faces(img_path=file_path, detector_backend="retinaface")
    if not faces:
        print("No faces found in the group photo.")
        return

    # Load the image and show thumbnails
    img = cv2.imread(file_path)
    print(f"\nFound {len(faces)} faces in the group photo.")
    print("Showing each face with a number. Enter names for each face when prompted.")
    
    # Display faces with numbers
    for idx, face in enumerate(faces, 1):
        region = face["facial_area"]
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        face_img = img[y:y+h, x:x+w]
        
        # Show face with number
        cv2.imshow(f"Face {idx}", face_img)
        cv2.moveWindow(f"Face {idx}", 100 + (idx-1)*200, 100)
        
    # Get names for each face
    for idx, face in enumerate(faces, 1):
        name = input(f"\nEnter name for face {idx} (or press Enter to skip): ").strip()
        if name:
            # Save face image
            region = face["facial_area"]
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            face_img = img[y:y+h, x:x+w]
            save_path = f"{name}_group_{idx}.jpg"
            cv2.imwrite(save_path, face_img)
            # Store in database
            store_embeddings(name, [save_path])
            print(f"Stored face {idx} as {name}")
    
    # Clean up windows
    for idx in range(1, len(faces) + 1):
        cv2.destroyWindow(f"Face {idx}")

def choose_files(multiple=True):
    root = tk.Tk(); root.withdraw()
    if multiple:
        return filedialog.askopenfilenames(title="Select images")
    else:
        return filedialog.askopenfilename(title="Pick test image")

def get_group_photo():
    method = input("Get group photo from (f)ile, (c)amera, or (q)uit? [f/c/q]: ").strip().lower()
    if method == 'q':
        return None
    elif method == 'c':
        save_path = "group_photo.jpg"
        return capture_from_camera(save_path)
    else:
        return choose_files(multiple=False)

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

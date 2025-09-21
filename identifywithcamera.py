import cv2
import os
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

def choose_files(multiple=True):
    root = tk.Tk(); root.withdraw()
    if multiple:
        return filedialog.askopenfilenames(title="Select images")
    else:
        return filedialog.askopenfilename(title="Pick test image")

if __name__ == "__main__":
    erase = input("Erase the database and start fresh? (y/n): ").strip().lower()
    if erase == 'y':
        erase_db()
    init_db()

    # Initial message only
    print("--- Face Database Setup ---")

    # Optionally add more images for existing or new people
    while True:
        # Always refresh and display known people and image counts before the prompt
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("SELECT person, COUNT(*) FROM faces GROUP BY person")
        known_people = c.fetchall()
        conn.close()
        if known_people:
            print("Known people in the database:")
            for idx, (pname, pcount) in enumerate(known_people, 1):
                print(f"  {idx}. {pname}  |  No of images available: {pcount}")
        else:
            print("No known people in the database yet.")
        person_input = input("Enter serial number or name to add images for (or just press Enter to finish): ").strip()
        if not person_input:
            break
        # Try to resolve serial number
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("SELECT person, COUNT(*) FROM faces GROUP BY person")
        known_people = c.fetchall()
        conn.close()
        name = None
        if person_input.isdigit() and known_people:
            idx = int(person_input) - 1
            if 0 <= idx < len(known_people):
                name = known_people[idx][0]
        if not name:
            name = person_input
        method = input("Add images from (f)ile or (c)amera? [f/c]: ").strip().lower()
        if method == 'c':
            save_path = f"{name}_cam.jpg"
            img_path = capture_from_camera(save_path)
            if img_path:
                store_embeddings(name, [img_path])
        else:
            files = choose_files(multiple=True)
            store_embeddings(name, files)

    while True:
        print("Now select a test image (or 'q' to quit)...")
        method = input("Test image from (f)ile, (c)amera, or (q)uit? [f/c/q]: ").strip().lower()
        if method == 'q':
            print("Exiting test phase.")
            break
        elif method == 'c':
            test_path = "test_cam.jpg"
            img_path = capture_from_camera(test_path)
            if not img_path:
                print("No image captured. Exiting test phase.")
                break
            identify(img_path)
        else:
            test = choose_files(multiple=False)
            if not test:
                print("No file selected. Exiting test phase.")
                break
            identify(test)

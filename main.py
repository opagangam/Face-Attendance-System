import os
from utils import get_img, find_faces_img, is_real_person, analyze_vid
from db import setup_db, record_attendance

media_dir = 'test_media'

def handle_file(path):
    if not os.path.exists(path):
        print("Can't find file:", path)
        return

    fname = os.path.basename(path).lower()
    
    # working with image
    if fname.endswith(".jpg") or fname.endswith(".jpeg") or fname.endswith(".png") or fname.endswith(".webp"):
        img = get_img(path)
        face_boxes = find_faces_img(img)
        seen = len(face_boxes)
        real = 0

        for box in face_boxes:
            t, r, b, l = box
            try:
                snippet = img[t:b, l:r]
                if is_real_person(snippet):
                    real += 1
            except:
                continue

        record_attendance(seen, real)
        print("[IMG]", fname, f"{seen} seen / {real} real")

    #  it's a video?
    elif fname.endswith(".mp4") or fname.endswith(".webm") or fname.endswith(".avi") or fname.endswith(".mov"):
        seen, real = analyze_vid(path)
        record_attendance(seen, real)
        print("[VID]", fname, f"{seen} seen / {real} real")

    else:
        print("File format not supported ->", fname)

def go_through_folder():
    if not os.path.isdir(media_dir):
        print("media folder missing.")
        return

    things = os.listdir(media_dir)
    if not things:
        print("Empty folder.")
        return

    print(f"Checking {len(things)} items in '{media_dir}'...\n")

    for item in things:
        full = os.path.join(media_dir, item)
        if os.path.isfile(full):
            handle_file(full)

if __name__ == '__main__':
    setup_db()
    go_through_folder()

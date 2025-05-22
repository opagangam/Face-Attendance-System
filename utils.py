# utils.py

import os
import cv2
import face_recognition
import numpy as np
import mediapipe as mp

#setting up FaceMesh to use  for liveliness
mesh_model = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Image loader 
def get_img(path):
    img = cv2.imread(path)
    return img

# Using face_recognition lib to find people in image
def find_faces_img(img):
    boxes = face_recognition.face_locations(img)
    return boxes

# Repeating here because sometimes video frame detection does not work properly
def find_faces_frame(f):
    return face_recognition.face_locations(f)

# Crude check to see if someone’s alive in the frame
def is_real_person(img):
    # convert to RGB bc Mediapipe needs it
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        res = mesh_model.process(rgb)
        if res.multi_face_landmarks:
            return True
    except Exception as err:
        print("FaceMesh failed:", err)

    return False

# Basic video analysis: count faces and whether they blink
def analyze_vid(p):
    vid = cv2.VideoCapture(p)
    seen = 0
    alive = 0

    while True:
        got, frame = vid.read()
        if not got:
            break

        these = find_faces_frame(frame)
        seen += len(these)

        for (t, r, b, l) in these:
            # pull the face region
            try:
                region = frame[t:b, l:r]
                if is_real_person(region):
                    alive += 1
            except:
                continue

    vid.release()
    return seen, alive

import queue
from deepface import DeepFace
from deepface.modules import verification, representation
import cv2
import numpy as np
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, VPacker
from scipy.spatial.distance import cosine
from datetime import datetime
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from multiprocessing import Lock
import threading

backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
metrics = ["cosine", "euclidean", "euclidean_l2"]


def process_frame(frame_data, db_embeddings, known_faces_only, chosen_model):
    """
    process_frame() will process a single frame and return the results
    only required for multiprocessing therefore outside the class
    :param frame_data: tuple containing frame, min_face_size, current_time, frame_count, total_frames
    :param db_embeddings: dictionary containing known embeddings
    :param known_faces_only: boolean flag to process only known faces
    :param chosen_model: chosen model for face recognition
    :return: results, new_embeddings, new_snapshots
    """
    frame, min_face_size, current_time, frame_count, total_frames = frame_data
    face_detector = cv2.FaceDetectorYN.create(
        model="face_detection_yunet_2023mar.onnx",
        config="",
        input_size=(320, 320),
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000,
        backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
        target_id=cv2.dnn.DNN_TARGET_CPU
    )
    height, width, _ = frame.shape
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(frame)
    faces = faces if faces is not None else []
    results = []
    new_embeddings = {}
    new_snapshots = {}

    for face in faces:
        x, y, w, h = map(int, face[:4])
        if w < min_face_size or h < min_face_size:
            continue

        face_img = frame[y:y + h, x:x + w]
        embedding = DeepFace.represent(img_path=face_img, model_name=chosen_model, enforce_detection=False)
        embedding = np.array(embedding[0]['embedding'])

        if known_faces_only:
            closest_match, min_distance = find_closest_match(embedding, db_embeddings)
            if min_distance <= 0.8:
                name = closest_match
            else:
                continue
        else:
            closest_match, min_distance = find_closest_match(embedding, db_embeddings)
            if min_distance <= 0.8:
                name = closest_match
            else:
                name = f"Person {len(db_embeddings) + 1}"
                new_embeddings[name] = embedding
                new_snapshots[name] = face_img

        results.append((frame_count, total_frames, frame, {'x': x, 'y': y, 'w': w, 'h': h}, name))

    return results, new_embeddings, new_snapshots


def find_closest_match(embedding, db_embeddings):
    """
    find_closest_match() will find the closest match for the given embedding
    :param embedding: embedding to compare
    :param db_embeddings: stored embeddings
    :return:
    """
    min_distance = float('inf')
    closest_match = None

    for name, db_embedding in db_embeddings.items():
        distance = verification.find_cosine_distance(embedding, db_embedding)
        if distance < min_distance:
            min_distance = distance
            closest_match = name

    return closest_match, min_distance


class SingleVidRecognizer:
    def __init__(self):
        """
        SingleVidRecognizer() will initialize the required variables and the face detector
        """
        self.known_faces_only = False
        self.csv_file = None
        self.SIMILARITY_THRESHOLD = 0.8
        self.FRAME_SKIP = 5  # Process every 5th frame
        self.RESIZE_FACTOR = 0.5  # Resize frame to 50% of original size
        self.CHOSEN_MODEL = models[0]  # VGG-Face
        self.CHOSEN_METRIC = metrics[0]  # Cosine
        self.CHOSEN_BACKEND = backends[0]  # opencv
        self.DISPLAY_SCALE = 1.5  # Scale factor for display
        self.BATCH_SIZE = 32
        self.USE_GPU = True if cv2.cuda.getCudaEnabledDeviceCount() > 0 else False

        self.num_processes = multiprocessing.cpu_count()
        self.manager = multiprocessing.Manager()
        self.db_embeddings = self.manager.dict()
        self.face_snapshots = self.manager.dict()
        self.person_count = multiprocessing.Value('i', 0)
        self.person_status = self.manager.dict()
        self.last_detection_time = self.manager.dict()
        self.db_embeddings_lock = Lock()
        self.face_snapshots_lock = Lock()

        self.frame_queue = queue.Queue(maxsize=30)
        self.process_event = threading.Event()

        # Initialize YuNet face detector
        self.face_detector = cv2.FaceDetectorYN.create(
            model="face_detection_yunet_2023mar.onnx",
            config="",
            input_size=(320, 320),
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )

    def set_known_faces_only(self, known_faces_only):
        """
        set_known_faces_only() will set the flag to process only known faces
        :param known_faces_only:
        :return:
        """
        self.known_faces_only = known_faces_only

    def set_csv_file(self, csv_file_path):
        """
        set_csv_file() will set the CSV file path for logging
        :param csv_file_path: path to the CSV file
        :return:
        """
        self.csv_file = csv_file_path
        with open(self.csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["PersonID", "StartTime", "EndTime", "Duration", "FrameCount"])

    def log_to_csv(self, person_id, start_time, end_time, duration, frame_count):
        """
        log_to_csv() will log the person's appearance to the CSV file
        :param person_id: person's ID
        :param start_time: start time of appearance
        :param end_time: end time of appearance
        :param duration: duration of appearance
        :param frame_count: total frames detected for the person during the appearance
        :return:
        """
        if self.csv_file:
            with open(self.csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([person_id, f"{start_time:.2f}", f"{end_time:.2f}", f"{duration:.2f}", f"{frame_count}"])

    def load_known_faces(self, folder_path):
        """
        load_known_faces() will load the known faces from the specified folder
        :param folder_path: path to the folder containing known faces
        :return:
        """
        with self.db_embeddings_lock, self.face_snapshots_lock:
            for image_name in os.listdir(folder_path):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, image_name)
                    embedding = DeepFace.represent(img_path=img_path,
                                                   model_name=self.CHOSEN_MODEL,
                                                   enforce_detection=False)
                    self.db_embeddings[os.path.splitext(image_name)[0]] = np.array(embedding[0]['embedding'])
                    # Store face snapshot
                    self.face_snapshots[os.path.splitext(image_name)[0]] = cv2.imread(img_path)

    def find_closest_match(self, embedding):
        """
        find_closest_match() will find the closest match for the given embedding
        :param embedding: embedding to compare
        :return:
        """
        min_distance = float('inf')
        closest_match = None
        with self.db_embeddings_lock:
            for name, db_embedding in self.db_embeddings.items():
                distance = verification.find_cosine_distance(embedding, db_embedding)
                if distance < min_distance:
                    min_distance = distance
                    closest_match = name
        return closest_match, min_distance

    def detect_faces(self, frame):
        """
        detect_faces() will detect faces in the given frame
        :param frame: input frame
        :return:
        """
        height, width, _ = frame.shape
        self.face_detector.setInputSize((width, height))
        _, faces = self.face_detector.detect(frame)
        return faces if faces is not None else []

    def get_multi_scale_embedding(self, face_img):
        """
        get_multi_scale_embedding() will calculate the embedding for the given face image
        :param face_img: input face image
        :return: average embedding of the face image at multiple scales
        """
        scales = [0.5, 1.0, 1.5]
        embeddings = []
        embedding_size = 2622
        for scale in scales:
            try:
                resized = cv2.resize(face_img, (0, 0), fx=scale, fy=scale)
                emb = DeepFace.represent(img_path=resized,
                                         model_name=self.CHOSEN_MODEL,
                                         enforce_detection=False)
                embeddings.append(np.array(emb[0]['embedding']))
            except Exception as e:
                print(f"Error in embedding calculation: {e}")
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(embedding_size)

    def is_better_snapshot(self, face_img):
        """
        is_better_snapshot() will check if the face snapshot is better than the existing one
        :param face_img: input face image
        :return: True if the snapshot is better, False otherwise
        """
        # Example of a simple brightness heuristic to ensure the snapshot isn't too dark
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        # Define a threshold for minimum brightness
        if brightness > 50:  # Adjust this threshold based on your video quality
            return True
        return False

    def update_person_status(self, name, current_time, face_img=None):
        """
        update_person_status() will update the status of the person based on the current frame
        :param name: name of the person
        :param current_time: current time in the video
        :param face_img: face image of the person
        :return:
        """
        # If the person was not present, mark them as present and start timing
        if name not in self.person_status:
            # First time seeing this person, add them to the dictionary
            self.person_status[name] = {
                'status': 'present',
                'start_time': current_time,
                'end_time': current_time,
                'detection_count': 1
            }
        elif self.person_status[name]['status'] == 'absent':
            # Person reappeared after being absent, update their status and reset start/end times
            person_info = self.person_status[name]
            person_info['status'] = 'present'
            person_info['start_time'] = current_time
            person_info['end_time'] = current_time
            person_info['detection_count'] = 1  # Reset detection count after reappearance
            # Reassign the whole dictionary entry to ensure the Manager recognizes the change
            self.person_status[name] = person_info
        else:
            # Person is still present, just update the end_time
            person_info = self.person_status[name]
            person_info['end_time'] = current_time
            person_info['detection_count'] += 1
            # Reassign the whole dictionary entry
            self.person_status[name] = person_info

        # Update the last detection time for the person
        self.last_detection_time[name] = current_time

        # After detecting a person for several frames (e.g., after 5 frames), update the face snapshot
        if face_img is not None and self.person_status[name]['detection_count'] > 5:
            # Optionally, you can also include a heuristic to check image quality (e.g., brightness)
            if self.is_better_snapshot(face_img):
                with self.face_snapshots_lock:
                    self.face_snapshots[name] = face_img

    def check_disappeared_faces(self, current_faces, current_time, frame_count):
        """
        check_disappeared_faces() will check for faces that have disappeared in the current frame
        :param current_faces: set of faces detected in the current frame
        :param current_time: current time in the video
        :param frame_count: current frame number
        :return:
        """
        # print("Current faces: ", current_faces)
        # Check for faces that have disappeared
        # print("People: ", self.person_status.keys())
        for name in list(self.person_status.keys()):
            if name not in current_faces and self.person_status[name]['status'] == 'present':
                # If the person has not been seen in the current frame, mark them as absent
                if current_time - self.last_detection_time[name] > 1:
                    # Log the duration they were present
                    start_time = self.person_status[name]['start_time']
                    end_time = self.person_status[name]['end_time']  # Last detected end time
                    duration = end_time - start_time
                    self.log_to_csv(name, start_time, end_time, duration, frame_count)

                    # Explicitly reassign the entire dictionary entry to ensure it's updated in the Manager
                    person_info = self.person_status[name]
                    person_info['status'] = 'absent'
                    self.person_status[name] = person_info  # Reassign the entire dictionary
                    # print(f"Updated status for {name}: {self.person_status[name]}")  # Debug print
                    # print(f"{name} marked as absent.")

    def run_recognition_algo(self, video_path, min_face_size):
        """
        run_recognition_algo() will run the face recognition algorithm on the given video
        :param video_path: path to the video file
        :param min_face_size: minimum face size for detection
        :return:
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print("Warning: FPS not detected correctly. Defaulting to 30 FPS.")
            fps = 30

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        print(f"Video FPS: {fps}, Total Frames: {total_frames}, Duration: {video_duration} seconds")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / fps

            if frame_count % self.FRAME_SKIP != 0:
                yield frame_count, total_frames, None, None, None
                continue

            faces = self.detect_faces(frame)
            results = []

            current_faces = set()

            for face in faces:
                x, y, w, h = map(int, face[:4])
                if w < min_face_size or h < min_face_size:
                    continue

                face_img = frame[y:y + h, x:x + w]
                embedding = self.get_multi_scale_embedding(face_img)

                if self.known_faces_only:
                    closest_match, min_distance = self.find_closest_match(embedding)
                    if min_distance <= self.SIMILARITY_THRESHOLD:
                        name = closest_match
                    else:
                        continue
                else:
                    closest_match, min_distance = self.find_closest_match(embedding)
                    if min_distance <= self.SIMILARITY_THRESHOLD:
                        name = closest_match
                    else:
                        with self.person_count.get_lock():
                            self.person_count.value += 1
                            name = f"Person {self.person_count.value}"
                        with self.db_embeddings_lock:
                            self.db_embeddings[name] = embedding
                        with self.face_snapshots_lock:
                            self.face_snapshots[name] = face_img

                self.update_person_status(name, current_time, face_img=face_img)
                results.append((frame_count, total_frames, frame, {'x': x, 'y': y, 'w': w, 'h': h}, name))
                current_faces.add(name)

            yield from results

            # Check for disappeared faces
            self.check_disappeared_faces(current_faces, current_time, frame_count)

        # After processing all frames, check for any remaining present faces
        current_time = frame_count / fps
        self.check_disappeared_faces(set(), current_time, frame_count)

        cap.release()

    def create_summary_report(self, csv_file_path):
        """
        create_summary_report() will create a summary report based on the CSV file
        :param csv_file_path: path to the CSV file
        :return:
        """
        df = pd.read_csv(csv_file_path)
        if df.empty:
            print("CSV file is empty. No summary report created.")
            return

        summary = df.groupby('PersonID').agg({
            'Duration': 'sum',
            'StartTime': 'count'
        }).reset_index()

        summary.rename(columns={'Duration': 'Cumulative Time (sec)', 'StartTime': 'Total Appearances'}, inplace=True)
        summary['Cumulative Time (sec)'] = summary['Cumulative Time (sec)'].apply(lambda x: round(x, 2))

        summary_file_name = os.path.splitext(csv_file_path)[0] + "_summary.csv"
        summary.to_csv(summary_file_name, index=False)
        print(f"Summary report saved to {summary_file_name}")

    def create_face_summary_graph(self, csv_file_path):
        """
        create_face_summary_graph() will create a face summary graph based on the CSV file
        :param csv_file_path: path to the CSV file
        :return:
        """
        df = pd.read_csv(csv_file_path)

        if df.empty:
            print("No faces detected in the video. Skipping graph creation")
            return

        summary = df.groupby('PersonID').agg({
            'Duration': 'sum',
            'StartTime': 'count'
        }).reset_index()

        summary.rename(columns={'StartTime': 'Appearances'}, inplace=True)
        summary = summary.sort_values(by='Duration', ascending=True)

        # Filter out PersonIDs without snapshots
        summary = summary[summary['PersonID'].isin(self.face_snapshots.keys())]

        if summary.empty:
            print("No faces detected in the video. Skipping graph creation")
            return

        fig, ax = plt.subplots(figsize=(15, 10))
        bars = ax.barh(summary['PersonID'], summary['Duration'], color='skyblue')

        for i, (name, duration, appearances) in enumerate(
                zip(summary['PersonID'], summary['Duration'], summary['Appearances'])):
            if name in self.face_snapshots:
                face_img = cv2.cvtColor(self.face_snapshots[name], cv2.COLOR_BGR2RGB)
                face_img = cv2.resize(face_img, (100, 100))
                im = OffsetImage(face_img)

                txt = TextArea(name, textprops=dict(ha='center', va='top', fontsize=8, color='black'))

                vpacker = VPacker(children=[im, txt], align="center", pad=0, sep=5)

                ab = AnnotationBbox(vpacker, (0, i), xybox=(-60, 0),
                                    frameon=False,
                                    xycoords='data',
                                    boxcoords="offset points",
                                    pad=0.5)
                ax.add_artist(ab)

        plt.subplots_adjust(left=0.4)
        ax.set_yticks([])

        for bar, duration, appearance in zip(bars, summary['Duration'], summary['Appearances']):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2,
                    f'{round(width,2)} sec | {appearance} appearances',
                    ha='left', va='center')

        ax.set_xlabel("Cumulative Runtime (seconds)")
        ax.set_title("Cumulative Runtime and Appearances per Person in the Video")
        ax.set_ylim(-0.5, len(summary) - 0.5)

        plt.tight_layout()
        graph_file_name = os.path.splitext(csv_file_path)[0] + "_graph.pdf"
        plt.savefig(graph_file_name, bbox_inches='tight')
        plt.close()
        print(f"Face summary graph saved to {graph_file_name}")


if __name__ == "__main__":
    print("This script is not meant to be run directly. Please use the GUI.")

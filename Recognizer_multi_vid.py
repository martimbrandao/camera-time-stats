from deepface import DeepFace
from deepface.modules import verification
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
metrics = ["cosine", "euclidean", "euclidean_l2"]


class MultiVidRecogniser:
    def __init__(self):
        """
        MultiVidRecogniser() will initialize the required variables and the face detector
        """
        self.target_embedding = None
        self.target_face_img = None
        self.SIMILARITY_THRESHOLD = 0.8
        self.FRAME_SKIP = 5  # Process every 5th frame for performance boost
        self.CHOSEN_MODEL = models[0]
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

    def load_target_face(self, face_path):
        """
        load_target_face() will load the target face image and its embedding
        :param face_path: path to the target face image
        :return:
        """
        self.target_embedding = self.get_embedding(face_path)

    def get_embedding(self, img_path):
        """
        get_embedding() will Get the embedding of the target face image
        :param img_path: path to the target face image
        :return: embedding of the target face
        """
        embedding = DeepFace.represent(img_path=img_path, model_name=self.CHOSEN_MODEL, enforce_detection=False)
        return np.array(embedding[0]['embedding'])

    def find_face_in_videos(self, video_paths, frame_ranges=None, progress_callback=None):
        """
        find_face_in_videos() will find the target face in the provided videos
        :param video_paths: list of paths to the video files
        :param frame_ranges: dictionary containing the time frames to process for each video
        :param progress_callback: callback function to update the progress
        :return: dictionary containing the occurrences of the target face in each video
        """
        results = {}
        for i, video_path in enumerate(video_paths):
            video_name = os.path.basename(video_path)
            if frame_ranges and video_name in frame_ranges:
                ranges = frame_ranges[video_name]
            else:
                ranges = [(0, None)]  # Process the entire video if no time frame is provided

            occurrences = self.process_video(video_path, ranges, i + 1, len(video_paths), progress_callback)
            results[video_name] = occurrences
        return results

    def detect_faces(self, frame):
        """
        detect_faces() will detect faces in the provided frame
        :param frame: input frame
        :return:
        """
        height, width, _ = frame.shape
        self.face_detector.setInputSize((width, height))
        _, faces = self.face_detector.detect(frame)
        return faces if faces is not None else []

    def get_multi_scale_embedding(self, face_img):
        """
        get_multi_scale_embedding() will get the embedding of the face image at multiple scales and return the average
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

    def process_video(self, video_path, time_ranges, video_num, total_videos, progress_callback=None):
        """
        process_video() will process the video to find the target face in the specified time ranges
        :param video_path: path to the video file
        :param time_ranges: list of time ranges to process
        :param video_num: current video number
        :param total_videos: total number of videos
        :param progress_callback: callback function to update the progress
        :return:
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        occurrences = []
        in_occurrence = False
        occurrence_start = None
        last_detection = None
        frames_since_last_detection = 0
        DETECTION_DELAY = self.FRAME_SKIP * 3  # Number of frames to wait before ending an occurrence

        # Process the video within the specified time ranges
        for start_time, end_time in time_ranges:
            range_start = int(start_time * fps) if start_time is not None else 0
            range_end = int(end_time * fps) if end_time is not None else total_frames

            cap.set(cv2.CAP_PROP_POS_FRAMES, range_start)

            for frame_count in range(range_start, range_end + 1, self.FRAME_SKIP):
                ret, frame = cap.read()
                if not ret:
                    break

                faces = self.detect_faces(frame)
                face_detected = False

                for face in faces:
                    x, y, w, h = map(int, face[:4])
                    face_img = frame[y:y + h, x:x + w]
                    embedding = self.get_multi_scale_embedding(face_img)

                    distance = verification.find_cosine_distance(embedding, self.target_embedding)

                    if distance <= self.SIMILARITY_THRESHOLD:
                        face_detected = True
                        print(f"Face detected at frame {frame_count}")
                        last_detection = frame_count
                        frames_since_last_detection = 0
                        if not in_occurrence:
                            in_occurrence = True
                            occurrence_start = frame_count
                        break  # Found a matching face, no need to check others

                if not face_detected:
                    frames_since_last_detection += self.FRAME_SKIP

                if in_occurrence and frames_since_last_detection >= DETECTION_DELAY:
                    print(f"Face no longer detected at frame {frame_count}, ending occurrence")
                    # Face has not been detected for DETECTION_DELAY frames, end the occurrence
                    in_occurrence = False
                    occurrences.append((occurrence_start / fps, last_detection / fps))
                    occurrence_start = None

                if progress_callback and frame_count % self.FRAME_SKIP == 0:
                    progress_callback(video_num, total_videos, frame_count - range_start, range_end - range_start)

        # Check if we're still in an occurrence at the end of the video
        if in_occurrence:
            occurrences.append((occurrence_start / fps, last_detection / fps))

        cap.release()
        return occurrences

    def create_summary_graph(self, results, output_file):
        """
        create_summary_graph() will create a summary graph of the target face appearances across videos
        :param results: dictionary containing the occurrences of the target face in each video
        :param output_file: path to save the summary graph
        :return:
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        video_names = list(results.keys())
        num_appearances = [len(occurrences) for occurrences in results.values()]
        print("Num appearances: ", num_appearances)

        bars = ax.bar(video_names, num_appearances)
        ax.set_xlabel('Videos')
        ax.set_ylabel('Number of Appearances')
        ax.set_title('Appearances of Target Face Across Videos')
        plt.xticks(rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height}', ha='center', va='bottom')

        # Add target face image
        if self.target_face_img is not None:
            target_face = cv2.cvtColor(self.target_face_img, cv2.COLOR_BGR2RGB)
            target_face = cv2.resize(target_face, (100, 100))
            imagebox = OffsetImage(target_face, zoom=0.5)
            ab = AnnotationBbox(imagebox, (1, max(num_appearances)), xybox=(0, 40),
                                xycoords='data', boxcoords="offset points",
                                box_alignment=(0.5, 1), pad=0.5, frameon=False)
            ax.add_artist(ab)

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"Summary graph saved to {output_file}")


def parse_frame_ranges(file_path):
    """
    parse_frame_ranges() will parse the time frame ranges from the provided file
    :param file_path: path to the time frame ranges file
    :return:
    """
    frame_ranges = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' -> ')
            if len(parts) == 2:
                video_name = parts[0]
                ranges = parts[1].split(', ')
                frame_ranges[video_name] = [tuple(map(float, r.split('-'))) for r in ranges]
    return frame_ranges


def main(face_path, video_paths, frame_ranges_file, progress_callback=None):
    """
    main() is the main function to run the face recognition algorithm on multiple videos
    :param face_path: path to the target face image
    :param video_paths: list of paths of the video files
    :param frame_ranges_file: path to the time frame ranges file
    :param progress_callback: callback function to update the progress
    :return:
    """
    print("Starting to process")
    recognizer = MultiVidRecogniser()
    recognizer.load_target_face(face_path)

    print("Recognizer loaded!")

    frame_ranges = parse_frame_ranges(frame_ranges_file) if frame_ranges_file else None

    # Find the face in the videos
    results = recognizer.find_face_in_videos(video_paths, frame_ranges, progress_callback)

    # Generate a summary graph
    recognizer.create_summary_graph(results, "summary_graph.png")


if __name__ == "__main__":
    print("This script is not meant to be run directly. Please use the GUI.")

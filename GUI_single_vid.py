import tkinter as tk
from tkinter import filedialog, messagebox
import os
import cv2
from PIL import Image, ImageTk
import threading
from Recognizer_single_vid import SingleVidRecognizer


class SingleVidGUI:
    def __init__(self):
        """
        SingleVidGUI() will initialize the GUI window and all the required variables
        """
        self.window = tk.Tk()
        self.window.title("Face Recognition - DeepFace")
        self.window.geometry("1200x800")

        self.images_folder = ""
        self.video_path = ""
        self.min_face_size = tk.StringVar(value="30")
        self.similarity_threshold = tk.StringVar(value="0.35")
        self.show_video = tk.BooleanVar(value=True)

        self.video_label = None
        self.cap = None
        self.frame_count_label = None
        self.create_widgets()
        
        self.recognizer = SingleVidRecognizer()

    def create_widgets(self):
        """
        create_widgets() will create all the GUI elements
        :return:
        """
        # Video display area
        self.video_label = tk.Label(self.window)
        self.video_label.place(relx=0.5, rely=0.4, anchor='center')

        # Known faces folder selection
        tk.Label(self.window, text="Known Faces Folder:").place(x=50, y=750)
        self.folder_path = tk.StringVar()
        tk.Entry(self.window, textvariable=self.folder_path, width=60).place(x=180, y=750)
        tk.Button(self.window, text="Browse", command=self.browse_folder).place(x=340, y=770)
        tk.Button(self.window, text="CLEAR", command=self.clear_folder_path).place(x=410, y=770)

        # Video file selection
        tk.Label(self.window, text="Video File:").place(x=670, y=750)
        self.video_path_var = tk.StringVar()
        tk.Entry(self.window, textvariable=self.video_path_var, width=60).place(x=740, y=750)
        tk.Button(self.window, text="Browse", command=self.browse_video).place(x=890, y=770)
        tk.Button(self.window, text="CLEAR", command=self.clear_video_path).place(x=960, y=770)

        # Frame count label
        self.frame_count_label = tk.Label(self.window, text="Frames: 0 / 0")
        self.frame_count_label.place(x=560, y=730)

        # Minimum face size input
        tk.Label(self.window, text="Minimum Face Size (reduce to detect small faces):").place(x=50, y=700)
        tk.Entry(self.window, textvariable=self.min_face_size, width=10).place(x=340, y=700)

        # Similarity threshold input
        tk.Label(self.window, text="Similarity Threshold (reduce to avoid mismatches):").place(x=50, y=725)
        tk.Entry(self.window, textvariable=self.similarity_threshold, width=10).place(x=340, y=725)
        
        # Show video checkbox
        tk.Checkbutton(self.window, text="Show Video", variable=self.show_video).place(x=850, y=700)

        # Start processing button
        tk.Button(self.window, text="Start Processing", command=self.start_processing).place(x=560, y=700)

    def browse_folder(self):
        """
        browse_folder() will open a file dialog to select the folder containing known faces
        :return:
        """
        folder_selected = filedialog.askdirectory()
        self.folder_path.set(folder_selected)
        self.images_folder = folder_selected

    def browse_video(self):
        """
        browse_video() will open a file dialog to select the video file
        :return:
        """
        video_file = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        self.video_path_var.set(video_file)
        self.video_path = video_file
        self.display_video_frame()

    def clear_folder_path(self):
        """
        clear_folder_path() will clear the selected folder path
        :return:
        """
        self.folder_path.set("")
        self.images_folder = ""

    def clear_video_path(self):
        """
        clear_video_path() will clear the selected video path
        :return:
        """
        self.video_path_var.set("")
        self.video_path = ""
        if self.cap:
            self.cap.release()
            self.video_label.config(image='')
            self.video_label.image = None

    def display_video_frame(self):
        """
        display_video_frame() will display the first frame of the selected video in the GUI
        :return:
        """
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.video_label.config(image=photo)
                self.video_label.image = photo
            self.cap.release()

    def show_detection_mode_popup(self):
        """
        show_detection_mode_popup() will display a popup message for the user to show the detection mode
        :return:
        """
        if self.images_folder:
            message = "The program will detect only known faces from the selected images folder."
        else:
            message = "The program will detect all faces in the video as they appear."
        messagebox.showinfo("Detection Mode", message)

    def start_processing(self):
        """
        start_processing() will start the face recognition process
        :return:
        """
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file.")
            return

        try:
            min_face_size = int(self.min_face_size.get())
        except ValueError:
            messagebox.showerror("Error", "Minimum face size must be a valid integer.")
            return
            
        try:
            similarity_threshold = float(self.similarity_threshold.get())
        except ValueError:
            messagebox.showerror("Error", "Similarity threshold must be a valid float number.")
            return

        if self.images_folder:
            if not os.path.exists(self.images_folder) or len(os.listdir(self.images_folder)) == 0:
                messagebox.showwarning("Warning", "The selected folder is empty or doesn't exist. All faces will be detected.")
                self.images_folder = ""
        
        # clear recognizer to start a new analysis
        self.recognizer.set_similarity_threshold(similarity_threshold)
        self.recognizer.clear()

        # Show detection mode popup
        self.show_detection_mode_popup()

        # Set up the recognizer
        if self.images_folder:
            self.recognizer.load_known_faces(self.images_folder)
            self.recognizer.set_known_faces_only(True)
        else:
            self.recognizer.set_known_faces_only(False)

        # Set up CSV file
        video_dir = os.path.dirname(self.video_path)
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        csv_file_name = f"{video_name}_log.csv"
        csv_file_path = os.path.join(video_dir, csv_file_name)
        self.recognizer.set_csv_file(csv_file_path)

        def process_video():
            """
            process_video() will run the face recognition algorithm on the selected video
            :return:
            """
            for frame_count, total_frames, frame, facial_area, name in self.recognizer.run_recognition_algo(
                    self.video_path, min_face_size):
                # Update frame count label
                self.frame_count_label.config(text=f"Frames: {frame_count} / {total_frames}")
                self.window.update_idletasks()

                if frame is not None and facial_area is not None and name is not None:
                    # Draw the facial area and name on the frame
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    if self.show_video.get():
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (640, 480))
                        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                        self.video_label.config(image=photo)
                        self.video_label.image = photo

            # After processing
            if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
                # Create summary report
                self.recognizer.create_summary_report(csv_file_path)
                # Create face summary graph
                self.recognizer.create_face_summary_graph(csv_file_path)
                messagebox.showinfo("Success", "Video processing completed. Summary report and graph have been saved.")
            else:
                messagebox.showinfo("Info", "No faces detected in the video. Skipping summary report and graph creation.")

        # Run processing in a separate thread to keep the GUI responsive
        threading.Thread(target=process_video, daemon=True).start()

    def run(self):
        """
        run() will start the GUI main loop
        :return:
        """
        self.window.mainloop()


if __name__ == "__main__":
    gui = SingleVidGUI()
    gui.run()

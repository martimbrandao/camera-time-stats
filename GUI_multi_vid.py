import tkinter as tk
from tkinter import filedialog, messagebox
import os
import threading
from Recognizer_multi_vid import main as process_videos


class MultiVidGUI:
    def __init__(self):
        """
        MultiVidGUI() will initialize the GUI window and all the required variables
        """
        self.window = tk.Tk()
        self.window.title("Face Recognition - DeepFace")
        self.window.geometry("1200x800")

        self.face_path = ""
        self.video_paths = []
        self.frame_ranges_file = ""
        self.create_widgets()

    def create_widgets(self):
        """
        create_widgets() will create all the GUI elements
        :return:
        """
        # Face image selection
        tk.Label(self.window, text="Target Face:").place(x=50, y=50)
        self.face_path_var = tk.StringVar()
        tk.Entry(self.window, textvariable=self.face_path_var, width=80).place(x=250, y=50)
        tk.Button(self.window, text="Browse", command=self.browse_face).place(x=750, y=45)

        # Video files selection
        tk.Label(self.window, text="Video Files:").place(x=50, y=100)
        self.video_listbox = tk.Listbox(self.window, width=80, height=10)
        self.video_listbox.place(x=250, y=100)
        tk.Button(self.window, text="Add Videos", command=self.add_videos).place(x=750, y=100)
        tk.Button(self.window, text="Clear Videos", command=self.clear_videos).place(x=850, y=100)

        # Time frames file selection
        tk.Label(self.window, text="Time Frames File (Optional):").place(x=50, y=300)
        self.time_frames_var = tk.StringVar()
        tk.Entry(self.window, textvariable=self.time_frames_var, width=80).place(x=250, y=300)
        tk.Button(self.window, text="Browse", command=self.browse_time_frames).place(x=750, y=295)

        # Start processing button
        tk.Button(self.window, text="Start Processing", command=self.start_processing).place(x=500, y=350)

        # Progress label
        self.progress_var = tk.StringVar(value="Ready to process")
        self.progress_label = tk.Label(self.window, textvariable=self.progress_var, font=("Arial", 12))
        self.progress_label.place(x=500, y=400)

    def browse_face(self):
        """
        browse_face() will open a file dialog to select the target face image
        :return:
        """
        face_file = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if face_file:
            self.face_path_var.set(face_file)
            self.face_path = face_file

    def add_videos(self):
        """
        add_videos() will open a file dialog to select multiple video files
        :return:
        """
        video_files = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        for video_file in video_files:
            if video_file not in self.video_paths:
                self.video_paths.append(video_file)
                self.video_listbox.insert(tk.END, os.path.basename(video_file))

    def clear_videos(self):
        """
        clear_videos() will clear the selected video files
        :return:
        """
        self.video_paths = []
        self.video_listbox.delete(0, tk.END)

    def browse_time_frames(self):
        """
        browse_time_frames() will open a file dialog to select the time frames file
        :return:
        """
        time_frames_file = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if time_frames_file:
            self.time_frames_var.set(time_frames_file)
            self.frame_ranges_file = time_frames_file

    def update_progress(self, video_num, total_videos, frame_count, total_frames):
        """
        update_progress() will update the progress label with the current processing status
        :param video_num: current video number
        :param total_videos: total number of videos
        :param frame_count: current frame number
        :param total_frames: total number of frames
        :return:
        """
        progress_text = f"Video {video_num}/{total_videos} - {frame_count}/{total_frames} frames"
        self.progress_var.set(progress_text)

    def start_processing(self):
        """
        start_processing() will start the face recognition process
        :return:
        """
        if not self.face_path:
            messagebox.showerror("Error", "Please select a target face image.")
            return

        if not self.video_paths:
            messagebox.showerror("Error", "Please select at least one video file.")
            return

        # Check if time frames file is provided; if not, create a default one
        if not self.frame_ranges_file:
            messagebox.showinfo("Info", f"No Time Frames File selected.\nWill process whole video!")

        def process():
            """
            process() will run the face recognition algorithm on the selected video
            :return:
            """
            try:
                print("Face Path:", self.face_path, "Video Paths:", self.video_paths, "Time Frames File:", self.frame_ranges_file)
                process_videos(self.face_path, self.video_paths, self.frame_ranges_file, self.update_progress)
                messagebox.showinfo("Success", f"Processing completed.\nResults saved")
                self.progress_var.set("Processing completed")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during processing: {str(e)}")
                self.progress_var.set("Error occurred during processing")

        threading.Thread(target=process, daemon=True).start()

    def run(self):
        """
        run() will start the GUI main loop
        :return:
        """
        self.window.mainloop()


if __name__ == "__main__":
    gui = MultiVidGUI()
    gui.run()

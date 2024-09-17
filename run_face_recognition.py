import sys
from GUI_single_vid import SingleVidGUI as SingleVideoGUI
from GUI_multi_vid import MultiVidGUI as MultiVideoGUI


def main():
    """
    main() will check the command line arguments and run the appropriate GUI
    :return:
    """
    if len(sys.argv) != 2 or sys.argv[1] not in ['1', '2']:
        print("Usage: python run_face_recognition.py [1|2]")
        print("1 - Run single video version")
        print("2 - Run multi-video version")
        sys.exit(1)

    if sys.argv[1] == '1':
        gui = SingleVideoGUI()
    else:
        gui = MultiVideoGUI()

    gui.run()


if __name__ == "__main__":
    main()

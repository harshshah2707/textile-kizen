from inference import TextileDefectDetector

if __name__ == "__main__":
    detector = TextileDefectDetector()
    try:
        detector.run_webcam()
    except Exception as e:
        print(f"Error starting webcam: {e}")
        print("Ensure a camera is connected and not used by another application.")

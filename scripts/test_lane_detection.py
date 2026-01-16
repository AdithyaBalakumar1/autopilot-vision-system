import cv2
import time

from src.perception.object_detection import ObjectDetector
from src.perception.lane_detection import detect_lanes

detector = ObjectDetector()


def draw_lanes(frame, lines):
    if lines is None:
        return frame

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame

if __name__ == "__main__":
    # Open input video FIRST
    video = cv2.VideoCapture("sample_road.mp4")
    prev_time = 0

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps_input = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        "output_autopilot_demo.mp4",
        fourcc,
        fps_input,
        (width, height)
    )

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break


        # FPS calculation
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time

        # Lane detection
        lines = detect_lanes(frame)
        output = draw_lanes(frame, lines)

        # Object detection
        detections = detector.detect(frame)
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = det["label"]

            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                output,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

        # FPS overlay
        cv2.putText(
            output,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.imshow("Perception Output", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()


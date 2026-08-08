import cv2


def get_video_info(video_path):
    """
    Return basic video information.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if fps <= 0:
            fps = 25.0

        duration = (
            frame_count / fps
            if frame_count > 0
            else 0.0
        )

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "width": width,
            "height": height
        }

    finally:
        cap.release()


def sample_frames(video_path, sample_rate=1.0):
    """
    Sample approximately `sample_rate` frames per second.

    Returns:
        List of OpenCV BGR NumPy frames.
    """

    if sample_rate <= 0:
        raise ValueError(
            "sample_rate must be greater than zero."
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    frames = []

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25.0

        frame_interval = max(
            int(round(fps / sample_rate)),
            1
        )

        frame_index = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_index % frame_interval == 0:
                frames.append(frame)

            frame_index += 1

    finally:
        cap.release()

    if not frames:
        raise ValueError(
            "No frames could be extracted from the video."
        )

    return frames
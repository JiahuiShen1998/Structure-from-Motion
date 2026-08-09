"""Sample frames from a video at a fixed interval.

The front of the pipeline: the fisheye footage is continuous video, but SfM wants a
set of stills with enough baseline between them. Every 10th frame was used for the
recorded run, turning a ~600-frame clip into ~60 images.

Input   a video file readable by OpenCV
Output  frame_00.jpg, frame_01.jpg, ... in the output directory

Choosing the interval is a trade-off. Too small and consecutive frames are nearly
identical: matching still succeeds but the triangulation angle is tiny, so the 3D
points are poorly constrained and the mapper may refuse the pair. Too large and the
viewpoint change between frames outruns SIFT's robustness, matches collapse, and the
reconstruction splits into disconnected models. 10 worked for handheld walking pace at
the capture frame rate; faster motion needs a smaller interval.

Original name: video_cut.py. The frame-selection logic is unchanged; this adds a CLI
and a zero-padding width that keeps filenames sortable past 100 frames.
"""

import argparse
import os

import cv2 as cv


def save_frames(video_source, output_folder, interval, prefix="frame_", width=2):
    """Write every `interval`-th frame of `video_source` into `output_folder`.

    Args:
        video_source: path to the video file.
        output_folder: directory to create and write into.
        interval: keep one frame out of every `interval`. See the module docstring for
            how to choose it.
        prefix: filename prefix.
        width: zero-padding width for the frame counter. The original used 2, which
            collides once more than 100 frames are saved; it is widened automatically
            if the video is long enough to need it.

    Returns:
        The number of frames written.
    """
    os.makedirs(output_folder, exist_ok=True)

    cap = cv.VideoCapture(video_source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_source}")

    fps = cap.get(cv.CAP_PROP_FPS)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {video_source}: {fps:.2f} fps, {total_frames} frames")

    expected = max(1, total_frames // max(1, interval))
    width = max(width, len(str(expected)))

    frame_count = 0
    save_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_count % interval == 0:
            name = os.path.join(output_folder, f"{prefix}{save_count:0{width}d}.jpg")
            cv.imwrite(name, frame)
            save_count += 1
        frame_count += 1

    cap.release()
    print(f"[INFO] read {frame_count} frames, wrote {save_count} to {output_folder}")
    return save_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample frames from a video at a fixed interval."
    )
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--out", required=True, help="Directory to write frames into")
    parser.add_argument("--interval", type=int, default=10,
                        help="Keep one frame out of every N (default: 10)")
    parser.add_argument("--prefix", default="frame_",
                        help="Output filename prefix (default: frame_)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.interval < 1:
        raise SystemExit("--interval must be at least 1")
    save_frames(args.video, args.out, args.interval, prefix=args.prefix)


if __name__ == "__main__":
    main()

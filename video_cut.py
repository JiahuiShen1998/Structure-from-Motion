import cv2 as cv
import os

def save_image(video_source, output_folder, interval):
    
    output_folder = os.path.join(os.path.dirname(video_source), output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # open video file
    cap1 = cv.VideoCapture(video_source)
    if not cap1.isOpened():
        print(f"cannot open: {video_source}")
        return

    # Acquire information from video
    fps = int(cap1.get(cv.CAP_PROP_FPS))  # Acquire fps
    total_frames = int(cap1.get(cv.CAP_PROP_FRAME_COUNT))  
    print(f"video information: FPS={fps}, Total frames={total_frames}")

    frame_count = 0
    save_count = 0

    while True:
        ret, frame = cap1.read()
        if not ret:
            print("Video read out.")
            break

        # 按间隔保存帧
        if frame_count % interval == 0:
            output_name = os.path.join(output_folder, f"frame_{save_count:02d}.jpg")
            cv.imwrite(output_name, frame)
            print(f"saved: {output_name}")
            save_count += 1

        frame_count += 1

    cap1.release()
    print(f"The extraction is complete, a total of {save_count} images have been saved.")

# 设定输入视频和输出文件夹
video_source = r"D:\AT_Master\Team_Project\recording\ParkingDeckStraight\114_ParkingDeckStraight_left.mp4"
output_folder = "ParkingDeckStraight_images_left"  
interval = 10  # Save one every 10 frames

save_image(video_source, output_folder, interval)

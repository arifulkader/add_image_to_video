import cv2
import numpy as np

def extract_white_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    white_frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if np.mean(frame) == 255:  # Check if the frame is completely white
            white_frames.append(frame_count)
        
        frame_count += 1

    cap.release()
    return white_frames

video_path = 'input_video.mp4'
white_frames = extract_white_frames(video_path)
print(white_frames)

from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips

def overlay_images_on_white_frames(video_path, white_frames, image_path, output_path):
    video = VideoFileClip(video_path)
    frames_per_second = video.fps
    
    # Load the image to overlay
    overlay_image = ImageClip(image_path).set_duration(1/frames_per_second)

    # Create a list to hold the video clips
    clips = []

    for i in range(int(video.duration * frames_per_second)):
        frame = video.get_frame(i / frames_per_second)
        
        if i in white_frames:
            clips.append(overlay_image.set_start(i / frames_per_second))
        else:
            frame_clip = ImageClip(frame, duration=1/frames_per_second).set_start(i / frames_per_second)
            clips.append(frame_clip)

    # Concatenate all clips
    final_video = concatenate_videoclips(clips, method='compose')
    final_video.write_videofile(output_path, codec="libx264")

image_path = 'overlay_image.png'
output_path = 'output_video.mp4'
overlay_images_on_white_frames(video_path, white_frames, image_path, output_path)

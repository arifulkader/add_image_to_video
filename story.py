import cv2
import numpy as np
import cvzone

def round_corners(img, radius):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (radius, radius), (w - radius, h - radius), 255, -1)

    # Create a mask for the rounded corners
    corner_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(corner_mask, (radius, radius), radius, 255, -1)
    cv2.circle(corner_mask, (w - radius, radius), radius, 255, -1)
    cv2.circle(corner_mask, (radius, h - radius), radius, 255, -1)
    cv2.circle(corner_mask, (w - radius, h - radius), radius, 255, -1)

    mask = cv2.bitwise_and(mask, cv2.bitwise_not(corner_mask))
    img[mask == 0] = 0
    return img

def round_shape(img,frame_width,frame_height):
    mask = np.zeros((img.shape[:2]), dtype=np.uint8)
    cv2.circle(mask, (int(frame_width/2), int(frame_height/4)), min(frame_width, int(frame_height/2))//2, 255, -1)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return masked_img

def resize_with_ratio(img,new_width,new_height):
    
    height, width, _ = img.shape
    aspect_ratio = width / height

    if aspect_ratio > new_width / new_height:
      new_height = int(new_width / aspect_ratio)
    else:
      new_width = int(new_height * aspect_ratio)

    resized_img = cv2.resize(img, (new_width, new_height))

    output_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    output_img[:, :] = 255

    x_offset = (new_width - resized_img.shape[1]) // 2
    y_offset = (new_height - resized_img.shape[0]) // 2

    output_img[y_offset:y_offset+resized_img.shape[0], x_offset:x_offset+resized_img.shape[1]] = resized_img

    return cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)

def fade_in(img, background, alpha):
  
  foreground, alpha_channel, _, _ = cv2.split(img)

  # Convert all arrays to float32 for consistent calculations
  foreground = foreground.astype(np.float32)
  alpha_channel = alpha_channel.astype(np.float32)
  background = background.astype(np.float32)
  alpha = np.float32(alpha)  # Ensure alpha is also a float32

  # Invert the alpha channel
  inverted_alpha = 255 - alpha_channel

  # Apply element-wise multiplication to blend the foreground and background
  faded_foreground = cv2.multiply(alpha * foreground, inverted_alpha)
  faded_background = cv2.multiply((1 - alpha) * background, inverted_alpha)

  # Combine the faded foreground and background
  return cv2.add(faded_foreground, faded_background).astype(np.uint8)  # Convert back to uint8 for display

def main():
  video_path = 'Scene 01 Source.mp4'
  image_paths = ['collage/pro_1.jpg','collage/pro_2.jpg', 'collage/pro_3.jpg',"collage/pro_4.jpg"]  # Replace with your image paths
  output_video_path = 'output/story3.mp4'

  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print("Error opening video file")
    return

  # Get video properties
  frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = cap.get(cv2.CAP_PROP_FPS)

  # Create output video writer
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Replace with your desired codec
  out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

  
  img = [cv2.imread(image_path, cv2.IMREAD_UNCHANGED) for image_path in image_paths]

  fade_in_progress = 0
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    if 2 <= current_time <= 5:
      # Overlay the image
        # resized_img_with_alpha_1 = cv2.cvtColor(cv2.resize(img[0],(frame_width,int(frame_height/2))), cv2.COLOR_BGR2BGRA)
        # round_frame = round_shape(resized_img_with_alpha_1,frame_width,frame_height)
        # frame = cvzone.overlayPNG(frame, round_frame,[0,0])

        fade_in_progress = min((current_time - 2) / 4, 1.0)

        reshape_img_1 = resize_with_ratio(img[0],frame_width-30,710)
        updated_frame = cv2.cvtColor(reshape_img_1,cv2.COLOR_BGR2BGRA)
        x_position =int((frame_width-reshape_img_1.shape[1])/2)

        # I want to apply the fade in function in here 
        # faded_image = fade_in(updated_frame, frame, fade_in_progress)

        frame = cvzone.overlayPNG(frame, updated_frame,[x_position,25])

        reshape_img_2 = resize_with_ratio(img[1],frame_width-30,710)
        updated_frame = cv2.cvtColor(reshape_img_2,cv2.COLOR_BGR2BGRA)
        x_position =int((frame_width-reshape_img_2.shape[1])/2)

         # I want to apply the fade in function in here 

        frame = cvzone.overlayPNG(frame, updated_frame,[x_position,710+25+25])

    if 8 <= current_time <= 12:
        reshape_img_3 = resize_with_ratio(img[2],frame_width-30,710)
        reshape_img_3 = round_corners(reshape_img_3,50)
        updated_frame = cv2.cvtColor(reshape_img_3,cv2.COLOR_BGR2BGRA)
        x_position =int((frame_width-reshape_img_1.shape[1])/2)
        frame = cvzone.overlayPNG(frame, reshape_img_3,[x_position,25])

        reshape_img_4 = resize_with_ratio(img[3],frame_width-30,710)
        updated_frame = cv2.cvtColor(reshape_img_4,cv2.COLOR_BGR2BGRA)
        x_position =int((frame_width-reshape_img_2.shape[1])/2)
        frame = cvzone.overlayPNG(frame, reshape_img_4,[x_position,710+25+25])
    
    out.write(frame)

  cap.release()
  out.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()

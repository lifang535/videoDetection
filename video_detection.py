import cv2
import os
import sys
import time
import numpy as np

sys.path.append('../protos')
import grpc
import video_service_pb2, video_service_pb2_grpc

import threading
import multiprocessing
from concurrent import futures

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import requests
from PIL import Image, ImageDraw, ImageFont

torch.cuda.empty_cache()

car_num = 0
person_num = 0

"""
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
"""
# 将模型和相关操作移动到 GPU 上
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
device = torch.device("cuda")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
# model = torch.nn.DataParallel(model).to(device)

# Load a font with a larger size
font_size = 16
font = ImageFont.truetype("fonts/times new roman.ttf", font_size)

car_timer = time.time()
person_timer = time.time()

def run():
    # opt_process = torch.compile(process)

    # Input video file path
    input_video_path = "../input_videos/video03.mp4"

    # Output directory for frames
    output_frames_dir = "output_frames"
    os.makedirs(output_frames_dir, exist_ok=True)

    # Output video file path
    output_video_path = "../output_videos/processed_video03s.mp4"

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame (example: apply a filter)
        # processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Example: Convert to grayscale
        processed_frame = process(frame)

        # Save the processed frame as a JPG file
        frame_filename = os.path.join(output_frames_dir, f"frame_{int(cap.get(1))}.jpg")
        cv2.imwrite(frame_filename, processed_frame)

        # Write the processed frame to the output video
        out.write(processed_frame)


    # Release the video capture and writer objects
    cap.release()
    out.release()

    # Combine the processed frames back into a video
    output_frames = [os.path.join(output_frames_dir, filename) for filename in os.listdir(output_frames_dir)]
    output_frames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    for frame_path in output_frames:
        frame = cv2.imread(frame_path)
        output_video.write(frame)

    # Release the output video writer
    output_video.release()

    # Clean up: Delete the processed frames
    for frame_path in output_frames:
        os.remove(frame_path)

    # Clean up: Delete the frames directory
    os.rmdir(output_frames_dir)

    print("Video processing complete.")

def process(frame):
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    # image = Image.open("images/photo02.jpg")
    image = Image.fromarray(frame)

    #inputs = processor(images=image, return_tensors="pt")
    #outputs = model(**inputs)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}  # 移动输入数据到 GPU
    with torch.no_grad():  # 执行模型推理，确保不会进行梯度计算
        outputs = model(**inputs)    

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
        )

        global car_timer, person_timer, car_num, person_num

        if time.time() - car_timer > 1:
            if model.config.id2label[label.item()] == "car":
                car_num += 1
                car_timer = time.time()
        
        if time.time() - person_timer > 5:
            if model.config.id2label[label.item()] == "person":
                person_num += 1
                person_timer = time.time()

        label_text = f"{model.config.id2label[label.item()]} {round(score.item(), 3)}"
        draw.rectangle(box, outline="blue", width=3)
        draw.text((box[0], box[1]), label_text, fill="yellow", font=font)

    processed_frame = np.array(image)

    return processed_frame

if __name__ == "__main__":
    # opt_run = torch.compile(run)
    run()

    print("car_num: ", car_num)
    print("person_num: ", person_num)

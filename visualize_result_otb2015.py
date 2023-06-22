import cv2
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-name_video", help="enter your video name here", required=True, default='Human3')
parser.add_argument("-type_tracker", choices=['MIL', 'MedianFlow', 'TLD', 'KCF', 'MOSSE', 'CSRT'], required=True)
args = parser.parse_args()

# Tên tệp video được lưu trữ
video_name = f"Demo{args.name_video}_{args.type_tracker}.avi"
# Khởi tạo đối tượng VideoWriter với các thông số của video đầu ra
path_dataset = 'DatasetOTB2015/'
frame_rate = 30
path = f'{path_dataset}/{args.name_video}/img/0001.jpg'
img = cv2.imread(path)
print(path)
image_size = (img.shape[1], img.shape[0])
print(image_size)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(video_name, fourcc, frame_rate, image_size)


iouScore = []
idx = 0
ans = []

data = args.name_video
folder_data = os.path.join(path_dataset, data)

folder_img = os.path.join(folder_data, 'img')
images = sorted(os.listdir(folder_img))
file_groundtruth = os.path.join(folder_data, 'groundtruth_rect.txt')
gt = open(file_groundtruth, 'r')
gt = gt.readlines()
# preprocessing groundtruth
gts = []
for i in gt:
    # parse by ,
    if ',' in i:
        x, y, w, h = map(int, i.split(','))
    else:
        x, y, w, h = map(int, i.split())
    gts.append((x, y, w, h))
print(data, len(gts), len(images))
first_frame_video = cv2.imread(os.path.join(folder_img, images[0]))
if args.type_tracker == 'CSRT':
    tracker = cv2.legacy.TrackerCSRT_create()
if args.type_tracker == 'MIL':
    tracker = cv2.legacy.TrackerMIL_create()
if args.type_tracker == 'MedianFlow':
    tracker = cv2.legacy.TrackerMedianFlow_create()
if args.type_tracker == 'TLD':
    tracker = cv2.legacy.TrackerTLD_create()
if args.type_tracker == 'KCF':
    tracker = cv2.legacy.TrackerKCF_create()
if args.type_tracker == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()
tracker.init(first_frame_video, gts[0])
cv2.rectangle(first_frame_video, (gts[0][0], gts[0][1]), (gts[0][0] + gts[0][2], gts[0][1] + gts[0][3]), (255, 0, 0), 2)
cv2.imshow("Hihi", first_frame_video)

ious = []
for i in range(1, min(len(images), len(gts))):
    frame = cv2.imread(os.path.join(folder_img, images[i]))
    ok, bboxPredict = tracker.update(frame)

    bboxTruth = gts[i]
    bboxPredict = (int(bboxPredict[0]), int(bboxPredict[1]), int(bboxPredict[2]), int(bboxPredict[3]))

    # Draw truth bbox using green color
    cv2.rectangle(frame, (bboxTruth[0], bboxTruth[1]), (bboxTruth[0] + bboxTruth[2], bboxTruth[1] + bboxTruth[3]), (0, 255, 0), 2)
    cv2.rectangle(frame, (10, 10), (20, 20), (0, 255, 0), 2)
    cv2.putText(frame, "Ground truth", (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.rectangle(frame, (10, 40), (20, 50), (0, 0, 255), 2)
    cv2.putText(frame, "Prediction", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "MinhTT - 21520064", (frame.shape[1] - 170, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(frame, "Status:", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    # Tracking successful
    if ok:
        # Draw predict bbox using red color
        cv2.rectangle(frame, (bboxPredict[0], bboxPredict[1]), (bboxPredict[0] + bboxPredict[2], bboxPredict[1] + bboxPredict[3]), (0, 0, 255), 2)
        # print(bboxTruth, bboxPredict)
        cv2.putText(frame, "Tracking", (70, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        img = cv2.resize(frame, image_size)
    else:
        ious.append(0)
        cv2.putText(frame, "Lost", (70, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    out.write(frame)
    cv2.imshow(data, frame)
    cv2.waitKey(1)

cv2.destroyAllWindows()

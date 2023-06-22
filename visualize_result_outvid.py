import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-name_video", help="enter your video name here", required=True, default='Human3')
parser.add_argument("-type_tracker", choices=['MIL', 'MedianFlow', 'TLD', 'KCF', 'MOSSE', 'CSRT'], required=True)
args = parser.parse_args()
name_vid = args.name_video.split('.')[0]
video_name = f"Demo{name_vid}_{args.type_tracker}.avi"

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(args.name_video)

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

success, img = cap.read()
bbox = cv2.selectROI("Tracking", img, False)
print(bbox, img)

image_size = (img.shape[1], img.shape[0])
print(image_size)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
frame_rate = 30
out = cv2.VideoWriter(video_name, fourcc, frame_rate, image_size)


tracker.init(img, bbox)

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 3, 1)
    cv2.putText(img, "Tracking", (70, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

while True:
    timer = cv2.getTickCount()
    success, img = cap.read()

    success, bbox = tracker.update(img)
    print((bbox))
    cv2.putText(img, "Status:", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if success:
        drawBox(img, bbox)
    else:
        cv2.putText(img, "Lost", (70, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(img, "FPS: " + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(img, "MinhTT - 21520064", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    out.write(img)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break

    try:
        cv2.imshow("Tracking", img)
    except:
        break

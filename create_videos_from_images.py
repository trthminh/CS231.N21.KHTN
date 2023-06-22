import cv2
import os

# Thư mục chứa các file ảnh
path = "img1/"

# Tên tệp video được lưu trữ
video_name = "output.avi"

# Khởi tạo đối tượng VideoWriter với các thông số của video đầu ra
frame_rate = 30
img = cv2.imread('img1/img0000.jpg')
image_size = (img.shape[1], img.shape[0])
print(image_size)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(video_name, fourcc, frame_rate, image_size)

# Lặp qua các file ảnh và ghi chúng vào tệp video
for filename in os.listdir(path):
    img_path = os.path.join(path, filename)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, image_size)
        out.write(img)

# Giải phóng tài nguyên
out.release()
cv2.destroyAllWindows()

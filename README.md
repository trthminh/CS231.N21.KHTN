# CS231.N21.KHTN
Nhập môn Thị giác máy tính - Introduction to Computer Vision
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

## Single Object Tracking
This is our final project for CS231 - Introduction to Computer Vision course

## Overview
* This project experiments with the CSRT vs KCF tracker from OpenCV on the OTB2015 dataset and compares its performance with other trackers in OpenCV.
* Additionally, the project allows for video demos on any selected video from the computer.
* Some demo results on the OTB2015 dataset can be viewed [slide](https://github.com/trthminh/CS231.N21.KHTN/blob/main/21520064_Single%20Object%20Tracking%20Report.pptx)

## Member
|**Student ID**| **Member**|**Email**|
|-----------|-----------|-----------|
|21520064|Truong Thanh Minh|21520064@gm.uit.edu.vn|

## Course information:
- **University**: University of Information Technology - VNUHCM UIT.
- **Faculty**: Computer Science
- **Semester**: 2
- **Year**: 2022 - 2023
- **Teacher**: Ph.D Mai Tien Dung

## Code

### Prerequisites

1. Clone the repo
    ```
   git clone https://github.com/trthminh/CS231.N21.KHTN.git
    ```
2. Download the dataset [here](https://drive.google.com/file/d/1FQ5zReW3SAbK5ABvhDrZqhqvT004ujJD/view?usp=drive_link/)
3. Install the python dependency packages.
    ```
    pip install -r requirements.txt
    ```

### Usage
Terminal command:
  If you want to run on video in OTB2015:
  ```
  python single_object_tracking_otb2015.py -name_video [name] -type_tracker [tracker]
  ```

  If you want to run on the video you upload from your computer:
  ```
  python single_object_tracking_outvid.py -name_video [name] -type_tracker [tracker]
  ```

  Where, `name` is name of the video, `tracker` is type of the tracker in OpenCV. For example, you can run 
  ```
  python single_object_tracking_outvid.py -name_video mot.mp4 -type_tracker KCF
  ```
or
```
python single_object_tracking_otb2015.py -name_video Human3 -type_tracker CSRT
```

  Additional:

  If you want to create a video from a list of images, you can run:
  python create_videos_from_images.py -folder_images [folder]

  Where folder is placed contain your images. For example, you can run: 
  ```
python create_videos_from_images.py -folder_images img1
```
  
## Contact:
If you have any questions, feel free to open issues or contact us at the email address above.

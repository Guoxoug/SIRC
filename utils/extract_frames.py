import cv2
import os
from argparse import ArgumentParser
import requests
parser = ArgumentParser()
parser.add_argument(
    "video_dir",
    type=str,
    help="directory where video files to be extracted are"
)

args = parser.parse_args()

if not os.path.exists(args.video_dir):
    os.mkdir(args.video_dir)

# download videos
# reference url
# http://www.depeca.uah.es/colonoscopy_dataset/Hyperplasic/hyperplasic_01/videos/NBI.mp4
url = "http://www.depeca.uah.es/colonoscopy_dataset"
tail = "videos/NBI.mp4"
lesions = [
    "Hyperplasic/hyperplasic",
    "Serrated/serrated",
    "Adenoma/adenoma"
]

for lesion in lesions:
    for i in range(1, 11):
        video_url = os.path.join(
            url, f"{lesion}_{i:02d}",tail
        )
        resp = requests.get(video_url)
        filename = os.path.join(
            args.video_dir, f"{lesion.split('/')[-1]}_{i:02d}_NBI.mp4"
        )
        with open(filename, "wb") as f:
            f.write(resp.content)
        print(f"video saved to: {filename}")



# paths and filename
video_paths = [
    (os.path.join(args.video_dir, file), file.replace(".mp4", "")) 
    for file 
    in os.listdir(args.video_dir) 
    if file.endswith(".mp4")
]

img_dir = os.path.join(args.video_dir, "images")
if not os.path.exists(img_dir):
    os.mkdir(img_dir)

# load downloaded videos and extract frames 
for path, filename in video_paths:
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(
            os.path.join(img_dir, f"{filename}_{count}.jpg"), 
            image
        )     # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

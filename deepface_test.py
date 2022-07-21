import time
import os

import cv2
from deepface.DeepFace import find

# root_dir = 'C:\\Users\\neuro_srv2\\Downloads\\photos'

# for subdir, dirs, files in os.walk(root_dir):
#     for file in files:
#         photo_path = os.path.join(subdir, file)


db = 'C:\\Users\\neuro_srv2\\Downloads\\photos\\'

img = 'C:\\Users\\neuro_srv2\\Downloads\\123.png'

start = time.time()
df = find(img_path=img, db_path=db, model_name='ArcFace', detector_backend='retinaface', enforce_detection=False)
print('Finding iteratevely took: ' + str(time.time() - start) + '\n')
print(df)

for frame in df.get('identity'):
    path = db + frame.split('\\')[-1].replace('/', '\\')
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    cv2.imshow('1', image)
    cv2.waitKey(1)
    time.sleep(5)

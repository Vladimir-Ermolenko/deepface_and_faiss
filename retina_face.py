import os
import time

import cv2
from deepface.detectors import FaceDetector
from deepface.DeepFace import detectFace

path = '/Users/voverm/Desktop/DC/files/'
directory = os.fsencode(path)
not_processed = []
total_retina = 0
total_OpenCV = 0
counter = 0

for file in os.listdir(directory):
    try:
        filename = os.fsdecode(file)
        img_path = path + filename

        font = cv2.FONT_HERSHEY_SIMPLEX
        org_opencv = (50, 30)
        org_retina = (20, 30)
        fontScale = 1
        color = (255, 255, 255)

        # Line thickness of 2 px
        thickness = 2

        start_retina = time.time()
        img_1 = cv2.cvtColor(detectFace(img_path=img_path, detector_backend='retinaface',
                                        enforce_detection=False, align=False), cv2.COLOR_BGR2RGB)
        total_retina += time.time() - start_retina
        img_1_text = cv2.putText(img_1, 'Retina face', org_retina, font,
                                 fontScale, color, thickness, cv2.LINE_AA)
        start_OpenCV = time.time()
        img_2 = cv2.cvtColor(detectFace(img_path=img_path, detector_backend='opencv',
                                        enforce_detection=False, align=False), cv2.COLOR_BGR2RGB)
        total_OpenCV += time.time() - start_OpenCV
        img_2_text = cv2.putText(img_2, 'OpenCV', org_opencv, font,
                                 fontScale, color, thickness, cv2.LINE_AA)

        images = cv2.hconcat([img_1, img_2])
        os.chdir('/Users/voverm/Desktop/faces/')
        cv2.imwrite(filename, images * 255)
        cv2.imshow('both', images)
        cv2.waitKey(1)
        time.sleep(1)
        counter += 1
        print('Processed image number ' + str(counter))
    except Exception:
        not_processed.append(filename)
        continue

# print('Total time with RetinaFace: ' + str(total_retina))
# print('Time per image with RetinaFace: ' + str(total_retina / counter))
# print('Total time with OpenCV: ' + str(total_OpenCV))
# print('Time per image with OpenCV: ' + str(total_OpenCV / counter))

# import time
#
# from deepface import DeepFace
# import cv2
#
# db = '/Users/voverm/Desktop/DC/files/'
#
# img = '/Users/voverm/Downloads/123.jpg'
#
# df = DeepFace.find(img_path=img, db_path=db, model_name='ArcFace', detector_backend='retinaface')
# print(df)
# for frame in df.get('identity'):
#     path = frame[frame.rfind('/') + 1:]
#     image = cv2.imread(db + path, cv2.IMREAD_COLOR)
#     cv2.imshow('1', image)
#     cv2.waitKey(1)
#     time.sleep(5)

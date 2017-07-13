import cv2
import numpy as np
from PIL import Image


def process_image(img, face_pos):
  if len(face_pos) == 0:
    # print('No face found!')
    return None, None
  mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8) #create mask with the same size as image, but only one channel. Mask is initialized with zeros
  cv2.grabCut(img, mask, tuple(face_pos[0]), np.zeros((1,65), dtype=np.float64), np.zeros((1,65), dtype=np.float64), 1, cv2.GC_INIT_WITH_RECT) #use grabcut algorithm to find mask of face. See grabcut description for more details (it's quite complicated algorithm)
  mask = np.where((mask==1) + (mask==3), 255, 0).astype('uint8') #set all pixels == 1 or == 3 to 255, other pixels set to 0
  img_masked = cv2.bitwise_and(img, img, mask=mask) #create masked image - just to show the result of grabcut
  #show images
  # cv2.imshow(title, mask)
  # cv2.imshow(title+' masked', img_masked)

  img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert image to hsv
  channels = [0,1]
  channels_ranges = [180, 256]
  channels_values = [0, 180, 0, 256]
  histogram = cv2.calcHist([img_hsv], channels, mask, channels_ranges, channels_values) #calculate histogram of H and S channels
  histogram = cv2.normalize(histogram, None, 0, 255, cv2.NORM_MINMAX) #normalize histogram

  dst = cv2.calcBackProject([img_hsv], channels, histogram, channels_values, 1) # calculate back project (find all pixels with color similar to color of face)
  # cv2.imshow(title + ' calcBackProject raw result', dst)

  ret, thresholded = cv2.threshold(dst, 25, 255, cv2.THRESH_BINARY) #threshold result of previous step (remove noise etc)
  # cv2.imshow(title + ' thresholded', thresholded)

  # cv2.waitKey(5000)
  #put partial results into one final image

  # row1 = np.hstack((img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), img_masked))
  row2 = np.hstack((img_hsv, cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR), cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)))
  # return np.vstack((row1, row2))
  return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)

def detect_sleeves(imagePath):
  haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #change it to path to face cascade - it's inside opencv folder

  img = cv2.imread(imagePath)
  face_pos = haar_cascade.detectMultiScale(img, 1.3, 5, cv2.CASCADE_FIND_BIGGEST_OBJECT)
  if len(face_pos) == 0: #if haar cascade failed to find any face, try again with different (more accurate, but slower) settings
      face_pos = haar_cascade.detectMultiScale(img, 1.1, 3, cv2.CASCADE_FIND_BIGGEST_OBJECT)
  face_only, result = process_image(img, face_pos)
  if len(face_pos)!= 0:
    # cv2.imwrite('result_' + path, result) #save the result
    # cv2.imwrite('face_only_' + path, face_only)
    size_skin = np.count_nonzero(result)
    size_face = np.count_nonzero(face_only)
    if round(size_skin/size_face,2) > 1.8 :
      return 'Short sleeves'
    else :
      return 'Long sleeves'
  else :
    return 'Undetermined'



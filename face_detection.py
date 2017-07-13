import cv2

def detect_face(imagePath):
  # Get user supplied values
  cascPath = "haarcascade_frontalface_default.xml"


  # Create the haar cascade
  faceCascade = cv2.CascadeClassifier(cascPath)

  # Create the haar cascade
  faceCascade = cv2.CascadeClassifier(cascPath)

  # Read the image
  image = cv2.imread(imagePath)

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Detect faces in the image
  faces = faceCascade.detectMultiScale(
     gray,
     scaleFactor=1.1,
     minNeighbors=5,
     minSize=(30, 30),
     flags = cv2.CASCADE_SCALE_IMAGE #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
  )

  print("Found {0} faces!".format(len(faces)))

  coordonates = [None]
  # Draw a rectangle around the faces
  for (x, y, w, h) in faces:
    # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    coordonates = [x,y,x+w,y+h]

  return coordonates

detect_face('chemise_4.jpg')

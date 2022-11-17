import cv2
import os

imagesPath = "/Users/fsociety/Projects/wednesday/photos"

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
for imageName in os.listdir(imagesPath):
    print(imageName)
    image = cv2.imread(imagesPath + "/" + imageName)
    faces = faceClassif.detectMultiScale(image, 1.1, 10)
    for (x, y, w, h) in faces:
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, (150, 150))
        names = imageName.split(" ")
        cv2.imwrite("faces/" + names[0] + "-" + str(count) + ".jpg", face)
        count += 1
        # cv2.imshow("Face", face)
        # cv2.waitKey(0)

#     cv2.imshow("Image", image)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()

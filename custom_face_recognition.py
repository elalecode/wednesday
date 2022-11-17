import cv2
import os
import face_recognition
import numpy


imageFacesPath = "/Users/fsociety/Projects/wednesday/faces"
imageFaceComparePath = "/Users/fsociety/Projects/wednesday/photos/270101009022M F_FR.jpg"
# imageFaceComparePath = "/Users/fsociety/Projects/wednesday/photos/fotosinmargen.jpeg"

facesEncodings = []
facesNames = []
for file_name in os.listdir(imageFacesPath):
    image = cv2.imread(imageFacesPath + "/" + file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    f_coding = face_recognition.face_encodings(image, known_face_locations=[(0, 150, 150, 0)])[0]
    facesEncodings.append(f_coding)
    facesNames.append(file_name.split("-")[0])



#######################################
# DETECTAR SI EXISTE UN ROSTRO PARECIDO

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

imageToCom = cv2.imread(imageFaceComparePath)
imageToCom = cv2.cvtColor(imageToCom, cv2.COLOR_BGR2RGB)
faces = faceClassif.detectMultiScale(imageToCom, 1.2, 10)
for (x, y, w, h) in faces:
    faceToSearch = imageToCom[y:y + h, x:x + w]
    faceToSearch = cv2.cvtColor(faceToSearch, cv2.COLOR_BGR2RGB)
    faceToSearchEncoding = face_recognition.face_encodings(faceToSearch, known_face_locations=[(0, w, h, 0)])[0]
    result = face_recognition.compare_faces(facesEncodings, faceToSearchEncoding)
    numpy_list = numpy.array(result)
    index_all = numpy.where(numpy_list == True)[0]
    print("Coincidencias:")
    for index in index_all:
        print(facesNames[index])
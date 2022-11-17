import cv2
# ------------ READ DNN MODEL ------------
# Model architecture
prototxt = "model/deploy.prototxt"
# Weights
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt, model)
# ------- READ THE IMAGE AND PREPROCESSING -------
# cap = cv2.VideoCapture("Images_videos/video_001.mp4")
vc = cv2.VideoCapture(0)
ret, frame = vc.read()
while True:
    if frame is not None:
        cv2.imshow("preview", frame)
    ret, frame = vc.read()
    height, width, _ = frame.shape
    frame_resized = cv2.resize(frame, (300, 300))
    # Create a blob
    blob = cv2.dnn.blobFromImage(frame_resized, 1.0, (300, 300), (104, 117, 123))
    # ------- DETECTIONS AND PREDICTIONS ----------
    net.setInput(blob)
    detections = net.forward()
    #print("detections.shape:", detections.shape)
    for detection in detections[0][0]:
        #print("detection:", detection)
        if detection[2] > 0.5:
            box = detection[3:7] * [width, height, width, height]
            x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(frame, "Conf: {:.2f}".format(detection[2] * 100), (x_start, y_start - 5), 1, 1.2, (0, 255, 255), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

vc.release()
cv2.destroyAllWindows()
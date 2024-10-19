import cv2
from ultralytics import YOLO
img_pth = "train/images/karyotype-8_bmp_jpg.rf.c44618565ac3267f62a1a065d7ac9933.jpg"
model = YOLO("../runs/detect/train6/weights/best.pt")
results = model(source=img_pth, conf=0.10)
res_plotted = results[0].plot()
#cv2.imshow("result", res_plotted)
cv2.imwrite("../result3.jpg", res_plotted)
cv2.waitKey(0)


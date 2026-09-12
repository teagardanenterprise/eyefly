import cv2, sys
cap = cv2.VideoCapture(0)
print("opened:", cap.isOpened())
if not cap.isOpened():
    sys.exit("FAIL: grant Terminal camera access in "
             "System Settings > Privacy & Security > Camera, restart Terminal")
ok, f = cap.read()
print("read:", ok, "shape:", f.shape if ok else None)
print("fps:", cap.get(cv2.CAP_PROP_FPS))
cap.release()

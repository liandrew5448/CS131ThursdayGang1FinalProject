import cv2

for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    ret, frame = cap.read()
    print(f"Camera {i}: opened={cap.isOpened()}, read={ret}")
    cap.release()
import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')

line_position = 550

algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detect = []
offset = 6
counter = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    contorno, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame, (25, line_position), (1200, line_position), (255, 0, 0), 3)    

    for (i, c) in enumerate(contorno):
        (x, y, w, h) = cv2.boundingRect(c)
        validar_contorno = (w >= 80) and (h >= 80)
        if not validar_contorno:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "VEHICLE" + str(i + 1), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.line(frame, (25, line_position), (1200, line_position), (0, 127, 255), 3)
        
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)
        for (x, y) in detect:
            if y < (line_position + offset) and y > (line_position - offset):
                counter += 1
            cv2.line(frame, (25, line_position), (1200, line_position), (0, 127, 255), 3)
            detect.remove((x, y))
            print("VEHICLE COUNTER: " + str(counter))
    
    cv2.putText(frame, "VEHICLE COUNTER: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 255), 5)
            
                

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()
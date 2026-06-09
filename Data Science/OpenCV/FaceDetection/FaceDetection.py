import cv2

face_cascad = cv2.CascadeClassifier('C:/Users/Acer/OneDrive/Desktop/Pradeep/Data Science/OpenCV/PHASE_8/haarcascade_frontalface_default.xml')
eye_cascad = cv2.CascadeClassifier('C:/Users/Acer/OneDrive/Desktop/Pradeep/Data Science/OpenCV/PHASE_8/haarcascade_eye.xml')
smile_cascad = cv2.CascadeClassifier('C:/Users/Acer/OneDrive/Desktop/Pradeep/Data Science/OpenCV/PHASE_8/haarcascade_smile (1).xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascad.detectMultiScale(gray,1.1,5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
    region_gray = gray[y:y+h , x:x+w]
    region_color = frame[y:y+h , x:x+w]
    
    """
    x - 100
    y - 150
    w - 80
    h - 80
    
    face start with - (100,150)
    face ko 'w' , w = 80 > 180 => 180 ma end huncha
    'h' = 80 > 230 => 230 ma end huncha
    """
    
    eyes = eye_cascad.detectMultiScale(region_gray,1.1,10)
    if len(eyes) > 0:
        cv2.putText(frame, 'Eyes Detected', (x,y -30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2 )
        
    smiles = smile_cascad.detectMultiScale(region_gray,1.5,15)
    if len(smiles) > 0:
        cv2.putText(frame, 'Smiles Detected', (x,y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2 )
        
    
    cv2.imshow("Smart Face Detector", frame)
    
    if cv2.waitKey(1) & 0xFF== ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
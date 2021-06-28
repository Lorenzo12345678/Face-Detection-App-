import cv2

#importiamo i dati allenati
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

#accesso alla fotocamera
cam = cv2.VideoCapture(0)

while True:
    #leggiamo il valore dell'acquisizione
    successful_read_frame, frame = cam.read()
    #convertiamo in bianco e nero
    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    #crea un array delle faccie riconosciute e disegna un rettangolo intorno ad esse
    faces = face_cascade.detectMultiScale(frame_grey)
    for (x,y,w,h) in faces:
            center = (x + w//2, y + h//2)
            frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
            #Riconosce i sorrisi e vi disegna un rettangolo intorno 
            smiles = smile_cascade.detectMultiScale(frame_grey, 1.8, 20)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(frame, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
                cv2.putText(frame,'You are smiling',(sx,sy), 1, 1,(255,255,255),1)
    #Mostra il risultato frame per frame
    cv2.imshow('Face Detection App',frame)
    #Indica ogni quanto mostrare il frame
    cv2.waitKey(1)

print('completed')

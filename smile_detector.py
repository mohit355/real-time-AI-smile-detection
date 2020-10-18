import cv2

#face and smile classifier
face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector=cv2.CascadeClassifier('haarcascade_smile.xml')
# grab webcam feed
webcam=cv2.VideoCapture(0)

while True:
        
    # read the current frame from webcam
    successful_frame_read,frame=webcam.read()
    
    # if there is an error
    if not successful_frame_read :
        break

    #change to gray image
    frame_grayscale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detect face first
    faces=face_detector.detectMultiScale(frame_grayscale,1.3,5)   # return a array of points


    # print(faces)

    # Run face detection within those faces
    for(x,y,w,h) in faces:

        #Draw a rectangle around the detected face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,200,50),4)  # top left and bottom left bgr thickness

        #getting the sub frame of face array using slicing
        the_face=frame[y:y+h , x:x+w]

        #change to gray scale
        face_grayscale=cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)

        #detecting smile in each faces only
        smiles=smile_detector.detectMultiScale(face_grayscale,1.7,20)  # scale-factor (blur)= 1.7  minNeighbour=20

    
        # #find all smiles in a face
        # for(x_,y_,w_,h_) in smiles:

        #     cv2.rectangle(the_face,(x_,y_),(x_+w_,y_+h_),(50,50,200),4)  # top left and bottom left bgr thickness

        #Label this face as smiling
        if len(smiles)>0:
            cv2.putText(frame,"Smiling",(x, y+h+40),fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN,color=(255,255,255))
        

    # show the current frame
    cv2.imshow("Why so serious? SMILE",frame)

    #show the frame until 'a' is not pressed
    if cv2.waitKey(33)== ord('a'):
        break
# clean up
webcam.release()
cv2.destroyAllWindows()
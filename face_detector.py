import cv2

# load some pre-trained data on face frontals from opencv (haar cascade algorithm) : 
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# capture footage from webcam:
webcam= cv2.VideoCapture(0)


#looping frame by frame over the webcam footage:
while True:

    # capture frame from webcam
    frame_read , frame =webcam.read()

    #greyscale the captured frame
    greyscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces with the help of coordinates
    face_coordintes=trained_face_data.detectMultiScale(greyscaled_frame)

    #draw rectangle over the original frame
    for (x,y,w,h) in  face_coordintes:
        cv2.rectangle(frame, (x,y),(x+w,y+h), (250,150,50), 4)
    
    #show the the face detection in real time
    cv2.imshow("real time face detector",frame)
    #frame refreshes after every one milliseconds
    key=cv2.waitKey(1)

    #pressing q or Q to break out from the infinite loop
    if key==81 or key==113:
        break




# Same face detection over a specific image:-

#Choose an image to detect faces in :
# img=cv2.imread("grpface1.jpg")
#making the image greyscale:
# greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces:
# face_coordintes=trained_face_data.detectMultiScale(greyscaled_img)
# Draw the rectange around the coloured img :
# for (x,y,w,h) in  face_coordintes:
    # cv2.rectangle(img, (x,y),(x+w,y+h), (250,150,50), 4)
#show the img (but it only shows the img for a split second and
#   then closes it to complete the code)
# cv2.imshow("Face Detector ", img)
#holds the photo till a key is pressed 
# cv2.waitKey()
# print ("code completed")

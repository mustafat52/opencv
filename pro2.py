import cv2 
import numpy as np
   

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale,scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver




''' The above written code is a complex code for stacking images of different dimensions.  '''


def empty(a):
    pass


'''Below is the code for the trackbox '''

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", (400, 240))
cv2.createTrackbar("Threshold1","Parameters", 133, 255, empty)   
cv2.createTrackbar("Threshold2","Parameters", 50, 255, empty)
cv2.createTrackbar("Area","Parameters", 5000, 50000, empty)


''' below function will give the contour lines'''

def contours(img, img_output):
    contour , heirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_output, contour,-1, (255,0,255), 3)

    for cnt in contour:
        area = cv2.contourArea(cnt)
        area_check = cv2.getTrackbarPos("Area", "Parameters")
        if area < area_check:
            cv2.drawContours(img_output, cnt,-1, (255,0,255), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            print(len(approx))

            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(img_con, (x,y), (x+w, y+h),(0,255,0),5)

            cv2.putText(img_con,"Points :" + str(len(approx)), (x+w+20,y+20),cv2.FONT_HERSHEY_COMPLEX,
                                     .7,(0,255,0),2)

            cv2.putText(img_con,"Area :" + str(int(area)), (x+w+20,y+50),cv2.FONT_HERSHEY_COMPLEX,
                                     .7,(0,255,0),2)                         



kernel = np.ones((5,5), np.uint8)

cap = cv2.VideoCapture(0)
frameWidth = 300
frameHeight = 300

print("Camera opened sucessfully")

cap.set(3,frameWidth)
cap.set(4,frameHeight)

while True:
    _, img = cap.read()

    img_con = img.copy()

    imgblur = cv2.GaussianBlur(img, (7,7), 1)
    img_gray = cv2.cvtColor(imgblur, cv2.COLOR_BGR2GRAY)


    Threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    img_canny = cv2.Canny(img_gray, Threshold1 ,Threshold2)
    img_dilation = cv2.dilate(img_canny, kernel, iterations=1)

    contours(img_dilation, img_con)

    


    imgStack = stackImages(0.8,([img,img_gray,img_canny],
                               [img_dilation,img_con,img_con]))

    #cv2.imshow("Video", img)
    #cv2.imshow("Blur Video", img_canny) 
    cv2.imshow("Result", imgStack)




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()

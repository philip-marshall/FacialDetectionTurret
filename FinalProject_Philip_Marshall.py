import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from scipy.signal import find_peaks
import serial
import time

def detectFaceFromSkin(contourss, bin_mask, frame, arduino):
    x,y,z = frame.shape
    rad = 0
    point_x = 0
    point_y = 0
    index = 0
    dataX = 0
    dataY = 0
    coords = ""
    i = 0
    for cont in contourss:
        (xx, yy), radius = cv2.minEnclosingCircle(cont)
        if (radius > rad):
            rad = radius
            point_x = xx
            point_y = yy
            index = i
        i = i + 1
    mask = np.zeros(bin_mask.shape, np.uint8)
    cv2.circle(mask,(int(point_x), int(point_y)), int(rad), (255), -1)

    newMask = np.bitwise_and(mask, bin_mask)

    point_x = int(point_x - rad)
    point_y = int(point_y - rad)
    width = int(rad * 2)
    crop = newMask[point_y:point_y + width, point_x:point_x + width]

    largestCircleTotal = crop.sum()
    maxLargest = crop.size

    if (len(contourss) == 0):
        print("empty")
    if (largestCircleTotal/maxLargest == np.nan or largestCircleTotal/ maxLargest < .3):
        contourss = np.delete(contourss, index)
        detectFaceFromSkin(contourss, bin_mask, frame, arduino)
    else:
        cv2.rectangle(frame, (point_x, point_y), (point_x + width, point_y + width), (0,255,0), 1)
        dataX = int((int(point_x + rad) / int(y)) * 180)
        dataY = int((int(point_y + rad) / int(x)) * 180)
        print("X: " + str(dataX) + " Y: " + str(dataY))
        #if (dataX > 0 and dataY > 0 and dataX < 180 and dataY < 180):
        coords = "X" + str(dataX) + "Y" + str(dataY)
        if (arduino):
            arduino.write(bytes(coords, 'utf-8'))
            


# set the webcam
video = cv2.VideoCapture(0)

# grab the arduino from the serial port
arduino = serial.Serial(port='COM3', baudrate=9600, timeout=.1)

while(True):
      
    # read each frame of the webcam (ret is simply a boolean value of if the frame was successfuly captured)
    ret, frame = video.read()
    x,y,z = frame.shape
    scale_factor = 2

    frame = cv2.resize(frame, (int(y/scale_factor), int(x/scale_factor)))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    LUVBadImg = frame.copy()
    #LUVBadImg = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
    # l = LUVBadImg[:, :, 0]

    # llength, lwidth = l.shape
    # lHisto = np.zeros(256)
    # for i in range(llength):
    #     for j in range(lwidth):
    #         lHisto[l[i,j]] = lHisto[l[i][j]] + 1

    # # automatically detect the threshold to cut out the histogram
    # kernal = np.array([4,0,0,0,4,0,0,0,4,0,0,0,4,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,-6,-6,-6,-6,-6,-6,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,4,0,0,0,4,0,0,0,4,0,0,0,4])
    # # The above kernal when convolved with the histogram will find the greatest pits in the histogram and the biggest one will be used as the threshold
    # threshold = np.convolve(lHisto, kernal, mode="same").argmax()

    # # Now that we know the threshold, we can clip the histogram there so anything above the threshold will be set to 0 thus deleteing the background
    # for i in range(256):
    #     if (i > threshold):
    #         lHisto[i] = 0

    # l = (l < threshold) * l
    # LUVBadImg[:,:,0] = l

    #LUVBadImg = cv2.cvtColor(LUVBadImg, cv2.COLOR_LUV2BGR)

    b, g, r = LUVBadImg[:, :, 0], LUVBadImg[:, :, 1], LUVBadImg[:, :, 2]

    dark_skin_mask = (r > 96) & (g > 40) & (b > 10) & ((frame.max() - frame.min()) > 15) & (np.abs(r-g) > 15) & (r > g) & (r > b)

    keyedDarkRed = np.multiply(r, dark_skin_mask)
    keyedDarkGreen = np.multiply(g, dark_skin_mask)
    keyedDarkBlue = np.multiply(b, dark_skin_mask)


    keyedDarkImg = cv2.merge([keyedDarkBlue, keyedDarkGreen, keyedDarkRed])

    # kernal = np.ones((5,5),np.uint8)
    # opening = cv2.morphologyEx(keyedDarkImg, cv2.MORPH_OPEN, kernal)
  

    # kernal = np.array(
    #     [[0,1,1,1,0],
    #     [1,1,1,1,1],
    #     [1,1,1,1,1],
    #     [1,1,1,1,1],
    #     [0,1,1,1,0]], np.uint8)
    kernal = np.ones((10,10), np.uint8)
    closing = cv2.morphologyEx(keyedDarkImg, cv2.MORPH_CLOSE, kernal)
    kernal = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(keyedDarkImg, cv2.MORPH_OPEN, kernal)

    #blobs = cv2.cvtColor(keyedDarkImg, cv2.COLOR_HSV2RGB)
    blobs = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)
    bin_mask = np.zeros(blobs.shape, np.uint8)

    for i in range(int(x/scale_factor)):
        for j in range(int(y/scale_factor)):
            if (blobs[i][j] > 0):
                bin_mask[i][j] = 1

    keyed_krisp = cv2.bitwise_and(frame_gray, frame_gray, mask=bin_mask)
    
    kernel = np.ones((5,5),np.uint8)/25
    blured = cv2.filter2D(keyed_krisp, -1, kernel)

    kernal = np.array(
        [[0,1,0],
        [1,-4,1],
        [0,1,0]])
    colorEdge = cv2.filter2D(blured, -1, kernal)

    # # # contours, _ = cv2.findContours(colorEdge,1,2)

    # # # for cont in contours:
    # # #     (xx, yy), radius = cv2.minEnclosingCircle(cont)
    # # #     if (radius > 5):
    # # #         cv2.circle(frame, (int(xx),int(yy)), int(radius), (0,255,0), 1)

    # frames = np.zeros(frame.shape, np.uint8)
    # frames = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # hull_list = []
    # for i in range(len(contours)):
    #     hull = cv2.convexHull(contours[i])
    #     hull_list.append(hull)
    
    # for i in range(len(contours)):
    #     cv2.drawContours(frames, hull_list, i, (255,255,0))

    contourss, _ = cv2.findContours(image=colorEdge, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    try:
        detectFaceFromSkin(contourss, bin_mask, frame, arduino)
    except:
        print("Error")
        arduino.close()


    ######arduino.write(bytes(str(point_x) + "," + str(point_y), 'utf-8'))
    time.sleep(0.05)
    
    # for cont in contours:
    #     (xx, yy), radius = cv2.minEnclosingCircle(cont)
    #     xx = int(xx)
    #     yy = int(yy)
    #     radius = int(radius)

    #     if (np.sqrt((xx - point_x)**2 + (yy - point_y)**2) < rad):
    #         cv2.circle(frame,(xx, yy), radius, (255,255,255), 1)

    # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    # masked = cv2.bitwise_and(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), mask)

    # kernal = np.array(
    #     [[0,1,0],
    #     [1,-4,1],
    #     [0,1,0]])
    # colorEdge = cv2.filter2D(masked, -1, kernal)

    # contours, _ = cv2.findContours(colorEdge,1,2)
    # for cont in contours:
    #     (ex, why), radius = cv2.minEnclosingCircle(cont)

    #     if (radius > 15):
    #         cv2.circle(frame,(int(ex), int(why)), int(radius), (255,255,255), 1)

    frame = cv2.resize(frame, (int(y), int(x)))
    # for cnt in contour:
    #     approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    #     #if len(approx) > 30:
    #     cv2.drawContours(frame,[cnt],0,(0,255,255),-1)
        # show the current frame
    cv2.imshow('frame', frame)


    
    # check to see if quit buttom was pressed (q)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(frame.shape)
        break
  
# Once done capturing we need to release the webcam
video.release()

# end program
cv2.destroyAllWindows()
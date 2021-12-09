# FacialDetectionTurret
This is my final project for a Computer Vision and Image Processing class. I am using numpy and openCV to detect faces in an image without the use of artificial intelligence or machine learning.

The main python file will detect skin colored objects in the scene and based on their edges, create a circle around the object with the minimum possible radius to enclose all the skin colored pixels. Then the largest circle is checked to see the percentage of skin colored pixels versus the total pixel count is above a threshold, if so this is the users face. Otherwise, the program assumes it is another part of the body and the next largest circle will be checked.

Once the coordinates for the face is found, these are fed to an arduino through serial communication where the string will be parsed and used to move the turret in the x and y axis.

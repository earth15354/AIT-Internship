import imghdr
import cv2 as cv
import numpy as np
import sys
import math

class Video(object):
    def __init__(self):
        # Sets up the video class which displays the video from the webcam
        capture = cv.VideoCapture(0)

        b = True

        while b == True:
            # Captures a frame from the webcam
            ret, frame = capture.read()

            if ret == True:
                # Mirror Image
                frame = cv.flip(frame, 1)

                # Converts to grayscale
                grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                # Converts to hsv/hsi
                hsi_color = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

                # Rescaling
                rescale = self.rescaleImage(frame,0.5,1)

                # Low Pass Filter / Blurring
                blur = cv.GaussianBlur(grayscale,(11,11),cv.BORDER_DEFAULT)

                # High Pass Filter / Edge Detection
                edges = cv.Canny(grayscale,65,120)

                # Dilation and Erosion
                dilated = cv.dilate(edges, (3,3), iterations = 1)
                eroded = cv.erode(edges, (2,2), iterations = 1)

                # Opening
                dilateFirst = cv.dilate(edges, (3,3), iterations = 1)
                opened = cv.erode(dilateFirst, (3,3), iterations = 1)

                # Closing
                erodeFirst = cv.erode(edges, (2,2), iterations = 1)
                closed = cv.dilate(erodeFirst, (2,2), iterations = 1)

                # Hough Transformation
                hough = self.houghTransform(edges, 125)

                # Contour Detection Using Canny
                slightBlur = cv.GaussianBlur(grayscale,(3,3),cv.BORDER_DEFAULT)
                blurCanny = cv.Canny(slightBlur, 65, 120)
                contours, heirarchies = cv.findContours(blurCanny, cv.RETR_LIST,
                    cv.CHAIN_APPROX_SIMPLE)
                
                # Contour Detection Using Threshold
                ret, thresh = cv.threshold(grayscale, 125, 255, cv.THRESH_BINARY)
                contours2, heirarchies2 = cv.findContours(thresh, cv.RETR_LIST,
                    cv.CHAIN_APPROX_SIMPLE)

                # Draw Contours
                blank = np.zeros(frame.shape, dtype='uint8')
                cv.drawContours(blank, contours, -1, (0,255,0), 1)
                
                # Displays images
                cv.imshow('Original Webcam', frame)
                cv.imshow('Grayscale', grayscale)
                cv.imshow('HSV', hsi_color)
                cv.imshow('Blurred', blur)
                cv.imshow('Edge Detection', edges)
                cv.imshow('Dilation', dilated)
                cv.imshow('Erosion', eroded)
                cv.imshow('Opening', opened)
                cv.imshow('Closing', closed)
                cv.imshow('Hough Transformation', hough)
                cv.imshow('Binary Image', thresh)
                cv.imshow('Contours', blank)

                # Wait 30 milliseconds and if "q" is pressed, close the window
                if cv.waitKey(30) & 0xFF == ord('q'):
                    b = False
            else:
                b = False

        capture.release()
        cv.destroyAllWindows()
        
    def rescaleImage(self, frame: imghdr, widthScale: int, heightScale: int) -> imghdr:
        """
        Rescales an image given a width and height scaling factor

        Parameters:
            frame              the image to rescale
            widthScale         the width scaling factor
            heightScale        the height scaling factor
        Returns:
            rescaled frame     the rescaled image
        """
        width = int(frame.shape[1] * widthScale)
        height = int(frame.shape[0] * heightScale)

        dimensions = (width,height)

        return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

    def houghTransform(self, frame: imghdr, threshold: int):
        lines = cv.HoughLines(frame, 1, np.pi / 180, threshold, None, 0, 0)
        cframe = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv.line(cframe, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

        return cframe

if __name__ == "__main__":
    # Creates a video object
    Video()

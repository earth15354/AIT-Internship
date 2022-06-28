import imghdr
import cv2 as cv
import numpy as np
import pytesseract as tes
import sys
from matplotlib import pyplot as plt

class OCR(object):

    def videoImage(self, videoSource: str):
        """
        Takes an image from a given video source (0 for web cam), one second after
        the video starts. Displays the original image, the image with words boxed,
        the image with characters boxed, and prints the words that appeared in the image

        Parameters:
            videoSource        the path to the video
        """
        capture = cv.VideoCapture(int(videoSource))
        cv.waitKey(1000)
        ret, frame = capture.read()
        capture.release()
        cv.imshow('Webcam Pic', frame)
        print("\n" + 'We found the following words in your image!' + "\n" + 
            tes.image_to_string(frame))

        cv.imshow("Characters", self.boxChars(frame.copy()))
        cv.imshow("Words", self.boxWords(frame.copy()))

        cv.waitKey(0)
        cv.destroyAllWindows

        # while True:
        #     # Captures a frame from the webcam
        #     ret, frame = capture.read()

        #     if ret == True:
        #         # Mirror Image
        #         frame_flip = cv.flip(frame, 1)

        #         # Converts to grayscale
        #         grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #         cv.imshow('Webcam', frame)

        #         frame_chars = frame.copy()
        #         frame_words = frame.copy()

        #         cv.imshow("Characters", self.boxChars(frame_chars))
        #         cv.imshow("Words", self.boxWords(frame_words))

        #         # Wait 30 milliseconds and if "q" is pressed, close the window
        #         if cv.waitKey(30) & 0xFF == ord('q'):
        #             break
        #     else:
        #         print("was not successful")
        #         break

    def stillImage(self, imageSource: str):
        """
        Displays an image from the given source, another with the circled words, 
        a third with the circled characters, and prints the words found in the image

        Parameters:
            imageSource        the path to the image
        """
        image = cv.imread(imageSource)
        cv.imshow('Your Image', image)

        print("\n" + 'We found the following words in your image!' + "\n" + 
            tes.image_to_string(image))

        cv.imshow("Characters", self.boxChars(image.copy()))
        cv.imshow("Words", self.boxWords(image.copy()))

        cv.waitKey(0)
        cv.destroyAllWindows

    def boxChars(self, img: imghdr) -> imghdr:
        """
        Given an image, returns the same image with all the ASCII characters boxed

        Parameters:
            img        the original image
        Returns:
            image      the same image with characters boxed
        """
        dimensions = img.shape
        #con = r'--oem 3 --psm 6 outputbase characters'
        boxes = tes.image_to_boxes(img)#, config=con)
        for b in boxes.splitlines():
            b = b.split(" ")
            x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
            cv.rectangle(img, (x, dimensions[0]-y), (w, dimensions[0]-h), (100,100,225), 1)
            cv.putText(img, b[0], (x, dimensions[0]-y+15), cv.FONT_HERSHEY_COMPLEX_SMALL,0.75,(50,50,225),1)
        
        return img

    def boxWords(self, img: imghdr) -> imghdr:
        """
        Given an image, returns the same image with all ASCII words boxed

        Parameters:
            img        the original image
        Returns:
            image      the same image with words boxed
        """
        #con = r'--oem 3 --psm 6 outputbase characters'
        boxes = tes.image_to_data(img)#, config=con)
        for i,b in enumerate(boxes.splitlines()):
            if i != 0:
                b = b.split()
                if len(b) == 12 and float(b[10]) > 50:
                    x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                    cv.rectangle(img, (x, y), (w+x, h+y), (100,100,225), 1)
                    cv.putText(img, self.roundWord(b[10]) + ": " + b[11], (x, y), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.6,(50,50,225), 1)
        return img

    def roundWord(self, word: str) -> int:
        """
        Given string that represents a float, returns the rounded number as an int

        Parameters:
            word        a string of a float
        Returns:
            integer     the rounded value of the float
        """
        return str(round(float(word), 2))

if __name__ == "__main__":
    # Waits for user to specify image or video
    type_input = input("Please specify if you are providing an image or video (write 'Image' or 'Video'): ")
    if type_input == "Image":
        # Waits for user to specify path to image
        path_input = input("Please specify the path to your image: ")
        try:
            my_ocr = OCR()
            my_ocr.stillImage(path_input)
        except Exception as e:
            print(e)
    elif type_input == "Video":
        # Waits for user to specify path to video
        path_input = input("Please specify the path to your video: ")
        try: 
            my_ocr = OCR()
            my_ocr.videoImage(path_input)
        except Exception as e: 
            print(e)
    else:
        print("Improper input")
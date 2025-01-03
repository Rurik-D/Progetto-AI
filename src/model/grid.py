import cv2
import numpy as np

class Grid:
    def __init__(self, imgPath):
        """
            Open the image from a given path, applies filters,
            detects contours and extracts the grid applying a 
            prospettic transformation.
        """
        self.rawImage = None
        self.resizedImage = None
        self.filteredImage = None
        self.approx = None
        self.srcPoints = None
        self.dstPoints = None
        self.warped = None
        self.isGrid = False

        try:
            self.rawImage = cv2.imread(imgPath)
            self.resizedImage = self.applySelectiveResize()
            self.filteredImage = self.applyFilters()
            self.approx = self.approxContours()

            if len(self.approx) == 4: 
                self.warped, self.side = self.prospTransform()
                self.isGrid = True
            else:
                return None
        except:
            return None

    def applySelectiveResize(self):
        SIZE_LIMIT = 800
        height, width = self.rawImage.shape[:2]
        larger_side = max(height, width)
        if larger_side>SIZE_LIMIT:
            scale_factor = SIZE_LIMIT/larger_side
            resized_img = cv2.resize(self.rawImage, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            return resized_img
        else:
            return self.rawImage
    
    def applyFilters(self):
        """
            Applies following filters:
                - gray scale
                - median blur
                - adaptive thresholding (get a binary image)
                - reverses the image
        """
        gray = cv2.cvtColor(self.resizedImage, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        medianBlur = cv2.medianBlur(blurred,5)
        th = cv2.adaptiveThreshold(medianBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 11, 2)
        _, binary = cv2.threshold(th, 128, 255, cv2.THRESH_BINARY_INV)

        return binary


    def approxContours(self):
        """
            Finds all outer edges in the binary image and 
            approximates them to save memory, then approximates
            the edges to reduce the number of points while
            maintaining the overall shape.
        """
        contours, _ = cv2.findContours(self.filteredImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        return approx


    def prospTransform(self):
        """
            Defines the target points for the perspective 
            transformation and applies it.
            Returns warped image and destination points.
        """
        points = self.approx.reshape(len(self.approx), 2)
        self.srcPoints = sortPoints(points)
        
        side = max([
            np.linalg.norm(self.srcPoints[0] - self.srcPoints[1]),
            np.linalg.norm(self.srcPoints[1] - self.srcPoints[2]),
            np.linalg.norm(self.srcPoints[2] - self.srcPoints[3]),
            np.linalg.norm(self.srcPoints[3] - self.srcPoints[0])
        ])
        
        self.dstPoints = np.array([
                    [0, 0],
                    [side - 1, 0],
                    [side - 1, side - 1],
                    [0, side - 1] ], dtype="float32")

        M = cv2.getPerspectiveTransform(self.srcPoints, self.dstPoints)
        warped = cv2.warpPerspective(self.resizedImage, M, (int(side), int(side)))

        return warped, side


def sortPoints(pts):
    """
        Order the points clockwise.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

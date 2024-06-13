import cv2
import numpy as np

class Grid:
    def __init__(self, imgPath):
        """
            Open the image from a given path, applies filters,
            detects contours and extracts the grid applying a 
            prospettic transformation.
        """
        self.rawImage = cv2.imread(imgPath)
        self.filteredImage = self.applyFilters()
        self.approx = self.approxContours()
        self.srcPoints = None
        self.dstPoints = None
        self.warped = None
        self.isGrid = False

        if len(self.approx) == 4:
            self.warped = self.prospTransform()
            self.isGrid = True


    def applyFilters(self):
        """
            Applies following filters:
                - gray scale
                - median blur
                - adaptive thresholding (get a binary image)
                - reverses the image
        """
        gray = cv2.cvtColor(self.rawImage, cv2.COLOR_BGR2GRAY)
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
        warped = cv2.warpPerspective(self.rawImage, M, (int(side), int(side)))

        return warped


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


def zoomCells(warped, dst_points):
    for point in dst_points.tolist():
        print(type(point))
        punto = (point[0],point[1])
        print(punto)
        cv2.circle(warped, punto, 5, (0, 255, 0), -1)
    
    rows, cols = warped.shape[:2]

    for x in range(0, rows-rows//9, rows//9):
        print(x)
        for y in range(0, cols-cols//9, cols//9):
            print(y)
            M = np.float32([[9, 0, -y*9], [0, 9, -x*9]])
            dst_image = cv2.warpAffine(warped, M, (cols, rows))
            cv2.imshow("image", dst_image)
            cv2.waitKey(0)




# cv2.namedWindow('Sudoku Grid Points', cv2.WINDOW_NORMAL)
# cv2.imshow('Sudoku Grid Points', getGrid('C:\\Users\\giuse\\Desktop\\Progetto-AI\\aug\\_288_6294564.jpeg')[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()







# # Trova le linee orizzontali e verticali
# gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray_warped, 50, 150, apertureSize=3)

# lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

# # Trova i punti di intersezione delle linee
# points = []
# for line1 in lines:
#     for line2 in lines:
#         x1, y1, x2, y2 = line1[0]
#         x3, y3, x4, y4 = line2[0]
#         denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
#         if denom != 0:
#             px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
#             py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
#             points.append((px, py))

# # Filtra i punti per ottenere solo i punti di intersezione della griglia
# grid_points = []
# for px, py in points:
#     if 0 <= px < side and 0 <= py < side:
#         grid_points.append((int(px), int(py)))
# # print(((int(approx[1][0][0])),int((approx[1][0][1]))))
# # print(grid_points[0])
# # Visualizza i punti di intersezione
# print(dst_points)
# for point in src_points:
#     print(type(point))
#     punto = ((int(point[0][0])),(int(point[0][1])))
#     print(punto)
#     cv2.circle(warped, punto, 5, (0, 255, 0), -1)

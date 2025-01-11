# *Solve*Doku
<img src="/docs/graphics/00%20Base/LogoSolveDoku.png" alt="SolveDoku Logo" width=30% height=30%>

Solving a Sudoku has never been easier thanks to *Solve*Doku!<br>
*Solve*Doku, indeed, has been created as your personal Sudoku solver.<br>
It will be sufficient to upload a photo of your Sudoku through the appropriate interface created for the program, and *Solve*Doku will take care of the rest, providing you with your Sudoku completely solved.

## How it works?

*Solve*Doku is structured in the following main steps:
1. The program will recognize in your uploaded picture the grid of the Sudoku and it will ensure that it is perfectly centered.
2. The recognized grid will be cleaned in such a way as to leave space only for the numbers contained within it.
3. A scrolling and processing will be carried out on all the cells, so that the numbers contained within them can be correctly recognized by the numerical recognition model.
4. The numerical recognition model, that is, a CNN combined with an MLP for classification, will be responsible for correctly labeling the content of each cell.
5. After completely digitizing the image of the sudoku, a backtracking algorithm will handle the actual resolution of the sudoku, providing a solution.
6. The program will finally create a new digital grid with the solution to the Sudoku that was provided in the image you uploaded.

## The Grid
In order to recognize the grid, the program will, first of all, apply some filters to the image uploaded:
1. **Greyscale**, to reduce dimensionality;
2. **Gaussian blur**, to reduce noise;
3. **Median filter blur**, to facilitate contours recognition;
4. **Thresholding**, to obtain a binary image with a greater usability.<br><br>
![Grid filters](/docs/graphics/02%20The%20Grid/02_The_Grid_01.png "Original-Greyscale-Gaussian-Median-Thresholding")

Thereafter, the program will:
1. Find an approximation of **all the external contours**;
2. Select the **biggest** contour;
3. Approximate the biggest contour to an array of **four vertices**;
4. Apply a **perspective transformation**, through the obtained points, to focus only the image of the grid.<br><br>
![Center grid](/docs/graphics/02%20The%20Grid/02_The_Grid_02.png "External contours - biggest contour - 4 vertices - perspective transformation")
 
## The Board

In order to remove the board, the program will:
1. Detect **horizontal** lines;
2. Detect **vertical** lines;
3. **Combine** the lines to create a mask;
4. Remove the grid lines through the **mask**;
5. Apply a **morphological closure** to uniform the image.<br><br>
<img src="/docs/graphics/03%20The%20Board/03_The_Board.png" alt="Board cleaning" width=50% height=50%>

## The Cells
The scrolling of the cells in the image actually utilizes an affine transformation, whose matrix increases the width and height of the image nine times (that is, the fixed number of rows or columns in the grid), mimicking a zoom effect, which is iterated across the entire image by updating the coordinate values of a point on the columns ($C$) and rows ($R$). The affine transformation matrix thus appears as follows:

$$\begin{bmatrix}
9 & 0 & C\\
0 & 9 & R
\end{bmatrix}$$

While the program scroll each of the 81 cells of the Sudoku, it applies on each cells the following filters, in order to make the digits recognizable by the model:
1. **Greyscale**, to reduce dimensionality;
2. **CLAHE**, to amplify, without an overamplification of the noise, the contrast in the image, enhancing the definitions of edges;
3. **Thresholding**, to obtain a binary image with a greater usability;
4. **Complement**, to adapt the image to those ones the model was trained with;
5. **Dilation**, to enhance the number legibility, increasing contours thickness.
<br><br>
![Cell filters](/docs/graphics/04%20The%20Cells/04_The_Cells.png "Original-Greyscale-CLAHE-Thresholding-Complement-Dilation")

## The Model
The dataset used to train the model is [*Printed Numerical Digits Image Dataset*](https://github.com/kaydee0502/printed-digits-dataset "Click here to consult the dataset").<br><br>
The model has been structured in two blocks: it begins with a **Convolutional Neural Network** (*CNN*) for the extraction of the main features from the input image and ends with a **Multilayer Perceptron** (*MLP*) for the digits classification.<br><br>
The CNN repeats three times the following structure, going from an output of **32** to one of **64** and finally to one of **128**:
- **Convolution**, to extract main features from input;
- **Batch normalization**. to stabilize and accelerate training;
- **ReLu activation function**, to introduce non-linearity;
- **Pooling**, to reduce input dimensionality and parameters, without losing image main features.

The MLP takes as input the output of 1152 neurons ($128*3*3$) of the CNN and repeats three times the following structure, going from an output of **256** to one of **128** and finally to one of **10**, that represent the 10 possible classification of the digits in the cell (digits from 1 to 9 and 0, namely the empty cell):
- **Linear Transformation**, to convert the multidimensional tensor of CNN in a unidimensional format suitable for MLP classification and to combine and remodel the input features to allow the model to learn deep relations in data;
- **ReLu activation function**, to introduce non-linearity;
- **Dropout**, to prevent overfitting, casually disabling a percentage of neurons (in this model 50%).

Approximately, following repeated tests, the model final training accuracy reaches 96%, while final test accuracy amounts to 98%.

## The Resolution Algorithm

At the end of cells procesessing, the program generates a matrix, which portrays the Sudoku.<br>
In order to resolve the Sudoku, the program applies a **backtracking** algorithm.
Essentially, the algorithm:
- Finds an empty cell.
- Tries to enter a number from 1 to 9.
  - If a number can be entered without breaking the rules, the function calls itself recursively to solve the rest of the grid.
  - If entering that number leads to a dead end, the number is removed and the next number is tried.

## The Solution

The solution is displayed as a brand new grid, where **color** distinguishes preset digits (in black) from the solution digits (in green).<br><br>
<img src="/docs/graphics/05%20The%20Solution/05_The_Solution.png" alt="Solution" width=30% height=30%>

## The GUI
The whole program is supported by a graphical user interface that follows the user in all the steps to resolve the Sudoku.<br><br>
![The GUI](/docs/graphics/06%20The%20GUI/06_GUI.png "Start-Load - Solve-Solution")

## External Libraries
In order to create *Solve*Doku the following libraries have been used:
- `OpenCV`
- `PyTorch`
- `Numpy`
- `Pandas`
- `tkinter`
- `customtkinter`
- `pillow`
- `time`

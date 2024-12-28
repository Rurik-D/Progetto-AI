import time
 

def is_valid(board, row, col, num):
    """
        Checks whether a number (cannot) be inserted into a given grid cell (row, col) 
        according to the Sudoku rules.
    """
    # Checks if 'num' is already in the line
    if num in board[row, :]:
        return False
    
    # Checks if 'num' is already in the column
    if num in board[:, col]:
        return False
    
    # Checks if 'num' is already in the 3x3 square
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    if num in board[start_row:start_row+3, start_col:start_col+3]:
        return False
    
    return True

def solve_sudoku(board, timestart):
    """
        Solves Sudoku using backtracking. Find an empty cell and try to enter a number from 1 to 9. 
        If a number can be entered without breaking the rules, the function calls itself recursively 
        to solve the rest of the grid. If entering that number leads to a dead end, the number is 
        removed and the next number is tried.
    """
    empty = find_empty(board)
    if time.time() - timestart >= 30:
        return False 
    
    if empty is None:
        return True  
    row, col = empty
    
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row, col] = num
            
            if solve_sudoku(board, timestart):
                return True
            
            board[row, col] = 0 # Backtracking
    return False

def find_empty(board):
    """
        finds an empty cell (denoted by 0) in the grid and returns its position as a tuple (row, col).
        If there are no empty cells, returns None.
    """
    for i in range(9):
        for j in range(9):
            if board[i, j] == 0:
                return (i, j)  # Returns the position of the empty cell
    return None




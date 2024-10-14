from grid import Grid
import digits
import os



def main():
    img_paths = os.listdir('images\\Sudoku')
    maxLength = len(max(img_paths, key=len))
    notFoundTxt = ""

    for path in img_paths:
        grid = Grid(path)

        if grid.isGrid:
            solved = digits.get_solved_sudoku(grid)
            if solved:
                if not digits.check_validity():
                    notFoundTxt += f"\n{path}".ljust(maxLength, "\terr: bs") # Bad Solution
            else:
                notFoundTxt += f"\n{path}".ljust(maxLength, "\terr: bdt") #Bad Digits Translation
        else:
            # Aggiungi al file non trovati
            notFoundTxt += f"\n{path}".ljust(maxLength, "\terr: gnf") # Grid Not Found





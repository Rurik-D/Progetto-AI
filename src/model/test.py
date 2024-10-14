from grid import Grid
import digits
import os



def main():
    img_paths = os.listdir('images\\Sudoku')
    maxLength = len(max(img_paths, key=len))
    notFoundTxt = ""

    for path in img_paths:
        path = os.path.abspath('.') + f"\\images\\Sudoku\\{path}"
        grid = Grid(path)

        if grid.isGrid:
            solved = digits.get_solved_sudoku(grid)
            if type(solved) != type(None):
                if digits.is_valid(solved):
                    print(f"\n{path}".ljust(maxLength, ' ') + "\tOK!")
                else:
                    notFoundTxt += f"\n{path}".ljust(maxLength, ' ') + "\terr: bs" # Bad Solution
                    print(f"\n{path}".ljust(maxLength, ' ') + "\terr: bs")
            else:
                notFoundTxt += f"\n{path}".ljust(maxLength, ' ') + "\terr: bdt" #Bad Digits Translation
                print(f"\n{path}".ljust(maxLength, ' ') + "\terr: bdt")

        else:
            # Aggiungi al file non trovati
            notFoundTxt += f"\n{path}".ljust(maxLength, ' ') + "\terr: gnf" # Grid Not Found
            print(f"\n{path}".ljust(maxLength, ' ') + "\terr: gnf")


    with open(os.path.abspath('.') + "\\src\\model\\not_found.txt", encoding='utf-8', mode='w') as f:
        f.write(notFoundTxt)

main()


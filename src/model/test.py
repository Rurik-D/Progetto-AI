from grid import Grid
import digits
import os



def main():
    img_paths = os.listdir('C:\\Users\\giuse\\Desktop\\Progetto-AI\\Images\\Sudoku')
    workTxt = ""

    for path in img_paths:
        path = f"C:\\Users\\giuse\\Desktop\\Progetto-AI\\Images\\Sudoku\\{path}"
        grid = Grid(path)

        print(f"Controllo griglia: {path}")

        if grid.isGrid:
            print("Griglia apposto")

            print("Calcolo soluzione...")
            solved, existsSol = digits.get_solved_sudoku(grid)    
            if existsSol:
                print("Soluzione presente")
                print("Controllo correttezza: ")
                if digits.is_valid(solved):
                    workTxt += f"C:\\Users\\giuse\\Desktop\\Progetto-AI\\Images\\Sudoku\\{path}\n"
                    print("OK!\n")
                else:
                    print("err: bs\n")
            else:
                print("err: bdt\n")

        else:
            # Aggiungi al file non trovati
            print("err: gnf\n")


    with open(os.path.abspath('.') + "\\src\\model\\found.txt", encoding='utf-8', mode='w') as f:
        f.write(workTxt)

main()


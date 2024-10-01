import numpy as np

def is_valid(board, row, col, num):
    # Controlla se 'num' è già nella stessa riga
    if num in board[row, :]:
        return False
    
    # Controlla se 'num' è già nella stessa colonna
    if num in board[:, col]:
        return False
    
    # Controlla se 'num' è già nel quadrato 3x3
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    if num in board[start_row:start_row+3, start_col:start_col+3]:
        return False
    
    return True

def solve_sudoku(board):
    empty = find_empty(board)
    if empty is None:
        return True  # Sudoku risolto
    row, col = empty
    
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row, col] = num
            
            if solve_sudoku(board):
                return True
            
            board[row, col] = 0  # Undo la scelta (backtracking)
    
    return False

def find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i, j] == 0:
                return (i, j)  # Ritorna la posizione vuota
    return None

def print_board(board):
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - -")
        
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            
            if j == 8:
                print(board[i, j])
            else:
                print(f"{board[i, j]} ", end="")

# Esempio di una griglia di Sudoku
data_str = """
9 2 7 0 0 0 6 0 4
0 8 3 6 0 7 9 2 0
0 5 6 0 0 0 0 7 0
0 0 0 3 2 8 0 0 0
2 0 0 0 6 0 0 0 3
3 0 0 4 7 5 0 0 0
0 1 9 0 0 0 0 6 0
6 3 2 7 0 9 1 4 0
7 0 0 0 0 6 0 0 9
"""

# Converti la stringa in una lista di liste
data_list = [list(map(int, row.split())) for row in data_str.strip().split('\n')]

# Converte la lista di liste in un array NumPy
board = np.array(data_list)
print("Griglia iniziale:")
print_board(board)
if solve_sudoku(board):
    print("\nGriglia risolta:")
    print_board(board)
else:
    print("Nessuna soluzione trovata.")

'''
is_valid(board, row, col, num): Questa funzione controlla se un numero (num) può essere inserito in una 
                                determinata cella della griglia (row, col) rispettando le regole del Sudoku.

solve_sudoku(board): Questa è la funzione principale che risolve il Sudoku usando il backtracking. 
                     Trova una cella vuota e prova a inserire un numero da 1 a 9. 
                     Se un numero può essere inserito senza violare le regole, 
                     la funzione chiama sé stessa ricorsivamente per risolvere il resto della griglia. 
                     Se inserire quel numero porta a un vicolo cieco, il numero viene rimosso 
                     e si prova con il numero successivo.

find_empty(board): Questa funzione trova una cella vuota (indicata con 0) nella griglia e restituisce 
                   la sua posizione come una tupla (row, col). 
                   Se non ci sono celle vuote, restituisce None.

print_board(board): Questa funzione stampa la griglia del Sudoku in un formato leggibile.

'''



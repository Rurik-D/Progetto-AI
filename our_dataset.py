import os
import csv

# path image folder
folder_random_img = 'C:\\Users\\giuse\\Desktop\\Progetto-AI\\data'
folder_sudoku_img = 'C:\\Users\\giuse\\Desktop\\Progetto-AI\\aug'
#output csv file
csv_output = 'our_dataset.csv'

#label random img
label_random_img = 0
label_sudoku_img = 1

#lists of all files 
random_file_names = os.listdir(folder_random_img)
sudoku_file_names = os.listdir(folder_sudoku_img)

# files with .jpeg exstension
random_images = [file for file in random_file_names if file.endswith('.jpg')]
sudoku_images = [file for file in sudoku_file_names if file.endswith('.jpeg')]

# Write the data to the CSV file
with open(csv_output, mode='w', newline='') as file_csv:
    writer = csv.writer(file_csv)
    
    # Write the CSV header
    writer.writerow(['filename', 'label'])
    
    # write data of random images
    for immagine in random_images:
        writer.writerow([immagine, label_random_img])
    
    # write data of sudoku images 
    for immagine in sudoku_images:
        writer.writerow([immagine, label_sudoku_img])
        
print(f"File CSV creato: {csv_output}")
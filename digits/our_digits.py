import os
import csv

# def rinomina_file(cartella):
#     for i in range(10):
#         files = os.listdir(os.path.join(cartella, i))
#         num_cartella = i
#         for indice, file in enumerate(files):
#             nuovo_nome = f"{num_cartella}_{indice}{os.path.splitext(file)[1]}"
#             print(nuovo_nome)
#             vecchio_percorso = os.path.join(cartella, file)
#             nuovo_percorso = os.path.join(cartella, nuovo_nome)
            
#             os.rename(vecchio_percorso, nuovo_percorso)



# def rinomina_file():
#     main_folder_path = 'C:\\Users\\giuse\\Desktop\\Progetto-AI\\digits\\assets'

#     subfolders = [f.path for f in os.scandir(main_folder_path) if f.is_dir()]

#     for i, subfolder in enumerate(subfolders):
#         files = [f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))]
        
#         for j, file in enumerate(files):
#             extension = os.path.splitext(file)[1]
            
#             new_name = f"{os.path.basename(subfolder)}_{j}{extension}"
            
#             os.rename(os.path.join(subfolder, file), os.path.join(subfolder, new_name))


# rinomina_file()
   
#output csv file
csv_output = 'our_dataset.csv'

images = []
for n in range(10):
    path = f'C:\\Users\\giuse\\Desktop\\Progetto-AI\\digits\\assets\\{n}'
    folder = os.listdir(path)
    images.append([file for file in folder if file.endswith(('.jpg', '.jpeg'))])

# Write the data to the CSV file
with open(csv_output, mode='w', newline='') as file_csv:
    writer = csv.writer(file_csv)
    
    # Write the CSV header
    writer.writerow(['filename', 'label'])
    print(len(images[1]))
    for n in range(10):
        for i in range(len(images[n])):
            #print(images[i])
            writer.writerow([images[n][i], n])   
  
print(f"File CSV creato: {csv_output}")
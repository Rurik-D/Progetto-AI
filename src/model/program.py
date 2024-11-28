import os

NEW_PATH = "C:\\Users\\giuse\\Desktop\\Progetto-AI\\Images\\correct-sudoku\\"


def main():
    txtLines = open("C:\\Users\\giuse\\Desktop\\Progetto-AI\\src\\model\\found.txt", encoding="utf8", mode="r").readlines()
    for line in txtLines:
        spostaFile(line)

def spostaFile(filePath):
    filePath = filePath[:-1]
    name = filePath.split('\\')[-1]
    newPath = NEW_PATH + name
    os.rename(filePath, newPath)
    

main()

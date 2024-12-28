import json
from os import path

class Language:
    """
        This class manages the current language used in the GUI.
    """
    def __init__(self):
        self.curLang = "en"
        self.langMap = {}
        self.__updateLanguageMap()

    def __updateLanguageMap(self):
        langFilePath = self.__getLanguagePath()
        with open(langFilePath, encoding='utf8', mode='r') as f:
            self.langMap = json.load(f)

    def __getLanguagePath(self):
        return path.abspath(".") + f"\\src\\resources\\json\\lang_{self.curLang}.json"

    def swapLanguage(self, scelta):
        
        if scelta == "Italian":
            self.curLang = 'it'
        else: 
            self.curLang = 'en'
        self.__updateLanguageMap()


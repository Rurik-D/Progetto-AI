import json
from os import path

class Language:
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

    def swapLanguage(self):
        self.curLang = 'en' if self.curLang == 'it' else 'it'
        self.__updateLanguageMap()


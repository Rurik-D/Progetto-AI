import json
from os import path

def load_language(lang):
    langFilePath = path.abspath(".") + f"\\src\\resources\\json\\lang_{lang}.json"
    with open(langFilePath, encoding='utf8', mode='r') as f:
        return json.load(f)

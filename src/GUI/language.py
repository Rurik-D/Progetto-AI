import json


def load_language(lang):
    with open(f'gui\\lang_{lang}.json', encoding='utf8', mode='r') as f:
        return json.load(f)

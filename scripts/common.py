from pathlib import Path

class_name_map = {
    '0': 'bread',
    '1': 'dairy',
    '2': 'dessert',
    '3': 'egg',
    '4': 'fried',
    '5': 'meat',
    '6': 'pasta',
    '7': 'rice',
    '8': 'seafood',
    '9': 'soup',
    '10': 'vegetables',
}

cwd = Path.cwd()

datasets_path = cwd / "datasets"
prannays_edibles_path = datasets_path / "prannays_edibles"

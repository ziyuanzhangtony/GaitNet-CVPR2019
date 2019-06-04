import os,pickle
import numpy as np
from PIL import Image

def center(win):
    win.update_idletasks()
    width = win.winfo_width()
    height = win.winfo_height()
    x = (win.winfo_screenwidth() // 2) - (width // 2)
    y = (win.winfo_screenheight() // 2) - (height // 2)
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))

def load_database(path=""):
    if os.path.isfile('database/data.pickle'):
        with open('database/data.pickle', 'rb') as handle:
            database = pickle.load(handle)
    else:
        database = {}
    return database

def imresize(image,height, width):
    image = np.array(Image.fromarray(image).resize((width, height), Image.ANTIALIAS))
    return image
"""
input/
    content.jpg: the original image you want to transfer
    style.jpg: the style image

src/
    train.py: generate the results

Usage: python run.py
"""
import os

os.system("python -m src.train")
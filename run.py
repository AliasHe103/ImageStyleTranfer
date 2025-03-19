"""
input/
    content.jpg: the original image you want to transfer
    style.jpg: the style image

src/
    train.py: generate the results

Usage: python run.py
"""
import os

epochs = 10
steps_per_epoch = 100
learning_rate = 0.001

os.system(f"python -m src.train -epochs {epochs} -steps {steps_per_epoch} -lr {learning_rate}")
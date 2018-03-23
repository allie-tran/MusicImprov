import os

import pygame

pygame.init()

files = os.listdir('generated')
file = 'generated/' + files[0]

pygame.mixer.music.load(file)
pygame.mixer.music.play()

from PIL import Image 
import os

os.chdir('/home/kamau/Final/images')

for file in os.listdir():
    image = Image.open(file)
    if image.mode == 'RGB':
        print('yes')
        

'''
for file in os.listdir():
    rgba_image = Image.open(file)
    rgb_image = rgba_image.convert('RGB')
    rgb_image.save(file)
'''
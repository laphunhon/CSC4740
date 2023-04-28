# Final Project: Export Data from .tff file

import os
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

# Specify the font file path
font_path = "times.ttf"

# Load the font file using FontTools
font = TTFont(font_path)

# Get the Unicode character set of the font
char_set = font.getBestCmap()
margin = 5
# Define the image size and font size
for i in range(8,73):
    image_size = (i+2*margin, i+2*margin)
    font_size = i
    # Loop through all the characters in the Unicode character set
    for char_code in char_set.keys():
        # 0 = 48; 9= 57; A=65; Z=90; a=97; z=122
        flag = False
        if char_code >= 48 and char_code <=57:
            flag = True
        if char_code >= 65 and char_code <=90:
            flag = True
        if char_code >= 97 and char_code <=122:
            flag = True
        if not flag:
            continue
        # Convert the character code to a Unicode string
        char_str = chr(char_code)
        
        # Create a new image and draw the character on it
        image = Image.new("RGB", image_size, (255, 255, 255))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, font_size)
        draw.text((margin, margin), char_str, font=font, fill=(0, 0,0))

        # Save the image as a PNG file
        file_name = f"{char_code}_{i}.png"       
        file_path = os.path.join("font", file_name)
        image.save(file_path)







import cv2
import numpy as np

def clamp_rgb_value(value):
    # Clamp the value to the range [0, 1]
    return min(1.0, max(0.0, value))

def RGBtoCIE(r, g, b):
    # Clamp each RGB channel individually
    clamped_r = clamp_rgb_value(r)
    clamped_g = clamp_rgb_value(g)
    clamped_b = clamp_rgb_value(b)
    
    return clamped_r, clamped_g, clamped_b

# Define empty lists and dictionaries
r, g, b = [], [], []
rgb = []
s = []
s2 = set()
d1 = {}
d2 = {}

img = cv2.imread('blue.jpg')

# Iterate over the pixels using enumerate
for row_idx, row in enumerate(img):
    for col_idx, pixel in enumerate(row):
        # Convert pixel to a tuple for dictionary keys
        pixel_tuple = tuple(pixel)

        if pixel_tuple not in s2:
            s2.add(pixel_tuple)
            rgb.append(pixel)
            temp = RGBtoCIE(pixel[2], pixel[1], pixel[0])
            r.append(temp[0])
            g.append(temp[1])
            b.append(temp[2])
            s.append(tuple(temp))
            d2[pixel_tuple] = tuple([r[-1], g[-1], b[-1]])
            d1[d2[pixel_tuple]] = 1
        else:
            d1[d2[pixel_tuple]] += 1

avg_r = sum(r) / len(r)
avg_g = sum(g) / len(g)
avg_b = sum(b) / len(b)
print('average of red  :', avg_r)
print('average of green: ', avg_g)
print('average of blue : ', avg_b)

# Find the mode of the CIE colors
mode_value = max(d1.values())
mode_colors = [key for key, value in d1.items() if value == mode_value]
print(f"Mode Colors: {mode_colors}")

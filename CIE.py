import cv2
import numpy as np

from colormath.color_objects import SpectralColor, XYZColor
from colormath.color_conversions import convert_color

def RGBtoCIE(x):
# Define the RGB values (range 0-255)
    R = x[2]
    G = x[1]
    B = x[0]

    # Normalize the RGB values to the range 0.0 to 1.0
    normalized_R = R / 255.0
    normalized_G = G / 255.0
    normalized_B = B / 255.0

    # Define the color matching functions for the CIE 1931 Standard Observer (2-degree, d50 illuminant)
    # These functions represent the spectral sensitivity of the human eye at different wavelengths.
    color_matching_functions = SpectralColor(
        spec_380nm=0.0, spec_390nm=0.0, spec_400nm=0.0, spec_410nm=0.0, spec_420nm=0.0,
        spec_430nm=0.0, spec_440nm=0.0, spec_450nm=0.0, spec_460nm=0.0, spec_470nm=0.0,
        spec_480nm=0.0, spec_490nm=0.0, spec_500nm=0.0, spec_510nm=0.0, spec_520nm=0.0,
        spec_530nm=0.0, spec_540nm=0.0, spec_550nm=0.0, spec_560nm=0.0, spec_570nm=0.0,
        spec_580nm=0.0, spec_590nm=0.0, spec_600nm=0.0, spec_610nm=0.0, spec_620nm=0.0,
        spec_630nm=0.0, spec_640nm=0.0, spec_650nm=0.0, spec_660nm=0.0, spec_670nm=0.0,
        spec_680nm=0.0, spec_690nm=0.0, spec_700nm=0.0, spec_710nm=0.0, spec_720nm=0.0,
        spec_730nm=0.0, spec_740nm=0.0, spec_750nm=0.0, spec_760nm=0.0, spec_770nm=0.0,
        spec_780nm=0.0, spec_790nm=0.0, spec_800nm=0.0, spec_810nm=0.0, spec_820nm=0.0,
        spec_830nm=0.0, observer='2', illuminant='d50'
    )

    # Create a SpectralColor object representing your RGB color
    spectral_rgb = SpectralColor(spec_660nm=normalized_R, spec_550nm=normalized_G, spec_470nm=normalized_B)

    # Convert to XYZ color space
    xyz_color = convert_color(spectral_rgb, XYZColor)

    # Extract the X, Y, and Z components
    X = xyz_color.xyz_x
    Y = xyz_color.xyz_y
    Z = xyz_color.xyz_z

    # Display the CIE 1931 XYZ values
    return [X,Y,Z]

import cv2
import numpy as np
def run(img):

    # Define empty lists and dictionaries
    r, g, b = [], [], []
    rgb = []
    s = []
    s2 = set()
    d1 = {}
    d2 = {}

    #img = cv2.imread('green.jpg')

    for i in img:
        for j in i:
            # Convert j to a tuple for dictionary keys
            j_tuple = tuple(j)

            if j_tuple not in s2:
                s2.add(j_tuple)
                rgb.append(j)
                temp = RGBtoCIE(j)
                r.append(temp[0])
                g.append(temp[1])
                b.append(temp[2])
                s.append(tuple(temp))
                d2[j_tuple] = tuple([r[-1], g[-1], b[-1]])
                d1[d2[j_tuple]] = 1
            else:
                d1[d2[j_tuple]] += 1



                
            

    avg_r = sum(r)/len(r)
    avg_g = sum(g)/len(g)
    avg_b = sum(b)/len(b)
    print('average of red  :',avg_r)
    print('average of green: ',avg_g)
    print('average of blue : ',avg_b)
    #print(max(d1.values()))
    m = max(d1.values())
    for key, value in d1.items():
        if value == m:
            print(f" mode--Key: {key}, Value: {value}")
    return [avg_r,avg_g,avg_b]




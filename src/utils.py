def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

def convert_color_space(image, color_space):
    if color_space == 'gray':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif color_space == 'hsv':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError("Unsupported color space: {}".format(color_space))
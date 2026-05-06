def rgb2rgba(rgb):
    """Take a row of RGB bytes, and convert to a row of RGBA bytes."""
    rgba = []
    for i in range(0, len(rgb), 3):
        rgba += rgb[i:i+3]
        rgba.append(255)

    return rgba
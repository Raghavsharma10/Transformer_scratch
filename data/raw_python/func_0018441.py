def mix(color1, color2, pos=0.5):
    """
    Return the mix of two colors at a state of :pos:

    Retruns color1 * pos + color2 * (1 - pos)
    """
    opp_pos = 1 - pos

    red = color1[0] * pos + color2[0] * opp_pos
    green = color1[1] * pos + color2[1] * opp_pos
    blue = color1[2] * pos + color2[2] * opp_pos
    return int(red), int(green), int(blue)
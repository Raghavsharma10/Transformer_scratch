def rgba_to_int(cls, red, green, blue, alpha):
        """
        Encodes the color as an Integer in RGBA encoding

        Returns None if any of red, green or blue are None.
        If alpha is None we use 255 by default.

        :return:    Integer
        :rtype:     int
        """
        red = unwrap(red)
        green = unwrap(green)
        blue = unwrap(blue)
        alpha = unwrap(alpha)
        if red is None or green is None or blue is None:
            return None
        if alpha is None:
            alpha = 255
        r = red << 24
        g = green << 16
        b = blue << 8
        a = alpha << 0
        rgba_int = r+g+b+a
        if (rgba_int > (2**31-1)):       # convert to signed 32-bit int
            rgba_int = rgba_int - 2**32
        return rgba_int
def find_closest(self, color):
        """
        Find the closest color in the system to the given rgb values.

        :param color: Tuple of r, g, b values (scaled to 255).
        :returns:     Tuple of name and rgb closest to the given color.
        """
        # Find distance between colors and find name based on closest color
        rgb = sRGBColor(*color)
        lab = convert_color(rgb, LabColor, target_illuminant='D65')
        min_diff = float("inf")
        min_name, min_color = "", ()
        for known_name, known_color in self.items():
            known_rgb = sRGBColor(*known_color)
            known_lab = convert_color(known_rgb, LabColor,
                                      target_illuminant='D65')
            diff = delta_e_cie1976(lab, known_lab)
            if min_diff > diff:
                min_diff = diff
                min_name = known_name
                min_color = known_color
        return min_name, min_color
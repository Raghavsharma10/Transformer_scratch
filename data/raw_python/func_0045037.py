def _polar_to_cartesian(cx, cy, r, theta):
        """
        :param cx: X coord of circle
        :param cy: Y coord of circle
        :param r: Radius of circle
        :param theta: Degrees from vertical, clockwise, in radians
        :return: (x, y)
        """
        return cx - r * math.sin(theta), cy - r * math.cos(theta)
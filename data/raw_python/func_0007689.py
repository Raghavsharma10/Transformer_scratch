def to_absolute(self, x, y):
        """
        Converts coordinates provided with reference to the center \
        of the canvas (0, 0) to absolute coordinates which are used \
        by the canvas object in which (0, 0) is located in the top \
        left of the object.

        :param x: x value in pixels
        :param y: x value in pixels
        :return: None
        """
        return x + self.size/2, y + self.size/2
def width(self, level):
        """
        Width at given level
        :param level:
        :return:
        """
        return self.x_at_y(level, reverse=True) - self.x_at_y(level)
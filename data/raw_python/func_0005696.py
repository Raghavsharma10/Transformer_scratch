def color(cls, value):
        """task value/score color"""
        index = bisect(cls.breakpoints, value)
        return colors.fg(cls.colors_[index])
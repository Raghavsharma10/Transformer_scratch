def set(cls, color):
        """
        Sets the terminal to the passed color.
        :param color: one of the availabe colors.
        """
        sys.stdout.write(cls.colors.get(color, cls.colors['RESET']))
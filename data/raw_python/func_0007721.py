def set_value(self, value: str):
        """
        Sets the value of the 7-segment display
        :param value: the desired value
        :return: None
        """

        self.clear()

        if '.' in value:
            self._segments['period'].configure(background=self._color)

        if value in ['0', '0.']:
            self._segments['a'].configure(background=self._color)
            self._segments['b'].configure(background=self._color)
            self._segments['c'].configure(background=self._color)
            self._segments['d'].configure(background=self._color)
            self._segments['e'].configure(background=self._color)
            self._segments['f'].configure(background=self._color)
        elif value in ['1', '1.']:
            self._segments['b'].configure(background=self._color)
            self._segments['c'].configure(background=self._color)
        elif value in ['2', '2.']:
            self._segments['a'].configure(background=self._color)
            self._segments['b'].configure(background=self._color)
            self._segments['g'].configure(background=self._color)
            self._segments['e'].configure(background=self._color)
            self._segments['d'].configure(background=self._color)
        elif value in ['3', '3.']:
            self._segments['a'].configure(background=self._color)
            self._segments['b'].configure(background=self._color)
            self._segments['g'].configure(background=self._color)
            self._segments['c'].configure(background=self._color)
            self._segments['d'].configure(background=self._color)
        elif value in ['4', '4.']:
            self._segments['f'].configure(background=self._color)
            self._segments['g'].configure(background=self._color)
            self._segments['b'].configure(background=self._color)
            self._segments['c'].configure(background=self._color)
        elif value in ['5', '5.']:
            self._segments['a'].configure(background=self._color)
            self._segments['f'].configure(background=self._color)
            self._segments['g'].configure(background=self._color)
            self._segments['c'].configure(background=self._color)
            self._segments['d'].configure(background=self._color)
        elif value in ['6', '6.']:
            self._segments['f'].configure(background=self._color)
            self._segments['g'].configure(background=self._color)
            self._segments['c'].configure(background=self._color)
            self._segments['d'].configure(background=self._color)
            self._segments['e'].configure(background=self._color)
        elif value in ['7', '7.']:
            self._segments['a'].configure(background=self._color)
            self._segments['b'].configure(background=self._color)
            self._segments['c'].configure(background=self._color)
        elif value in ['8', '8.']:
            self._segments['a'].configure(background=self._color)
            self._segments['b'].configure(background=self._color)
            self._segments['c'].configure(background=self._color)
            self._segments['d'].configure(background=self._color)
            self._segments['e'].configure(background=self._color)
            self._segments['f'].configure(background=self._color)
            self._segments['g'].configure(background=self._color)
        elif value in ['9', '9.']:
            self._segments['a'].configure(background=self._color)
            self._segments['b'].configure(background=self._color)
            self._segments['c'].configure(background=self._color)
            self._segments['f'].configure(background=self._color)
            self._segments['g'].configure(background=self._color)
        elif value in ['-']:
            self._segments['g'].configure(background=self._color)

        else:
            raise ValueError('unsupported character: {}'.format(value))
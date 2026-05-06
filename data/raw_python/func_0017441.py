def is_header(self):
        """
        Whether or not the cell is a header

        Any header cell will have "=" instead of "-" on its border.

        For example, this is a header cell::

            +-----+
            | foo |
            +=====+

        while this cell is not::

            +-----+
            | foo |
            +-----+

        Returns
        -------
        bool
            Whether or not the cell is a header
        """
        bottom_line = self.text.split('\n')[-1]

        if is_only(bottom_line, ['+', '=']):
            return True

        return False
def bottom_sections(self):
        """
        The number of cells that touch the bottom side.

        Returns
        -------
        sections : int
            The number of sections on the top
        """
        bottom_line = self.text.split('\n')[-1]
        sections = len(bottom_line.split('+')) - 2

        return sections
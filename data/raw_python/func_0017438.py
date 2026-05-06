def right_sections(self):
        """
        The number of sections that touch the right side.

        Returns
        -------
        sections : int
            The number of sections on the right
        """
        lines = self.text.split('\n')
        sections = 0
        for i in range(len(lines)):
            if lines[i].endswith('+'):
                sections += 1
        return sections - 1
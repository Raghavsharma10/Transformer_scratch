def top_sections(self):
        """
        The number of sections that touch the top side.

        Returns
        -------
        sections : int
            The number of sections on the top
        """

        top_line = self.text.split('\n')[0]
        sections = len(top_line.split('+')) - 2

        return sections
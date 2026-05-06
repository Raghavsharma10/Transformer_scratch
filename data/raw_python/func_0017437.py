def left_sections(self):
        """
        The number of sections that touch the left side.

        During merging, the cell's text will grow to include other
        cells. This property keeps track of the number of sections that
        are touching the left side. For example::

                        +-----+-----+
            section --> | foo | dog | <-- section
                        +-----+-----+
            section --> | cat |
                        +-----+

        Has 2 sections on the left, but 1 on the right

        Returns
        -------
        sections : int
            The number of sections on the left
        """
        lines = self.text.split('\n')
        sections = 0

        for i in range(len(lines)):
            if lines[i].startswith('+'):
                sections += 1
        sections -= 1

        return sections
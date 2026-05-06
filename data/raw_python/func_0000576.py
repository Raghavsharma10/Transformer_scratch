def mark_bl(self) -> int:
        """
        Mark unprocessed lines that have no content and no string nodes
        covering them as blank line BL.

        Returns:
            Number of blank lines found with no stringy parent node.
        """
        counter = 0
        stringy_lines = find_stringy_lines(self.node, self.first_line_no)
        for relative_line_number, line in enumerate(self.lines):
            if relative_line_number not in stringy_lines and line.strip() == '':
                counter += 1
                self.line_markers[relative_line_number] = LineType.blank_line

        return counter
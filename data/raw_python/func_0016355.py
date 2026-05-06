def line_statuses(self, filename):
        """
        Return a list of tuples `(lineno, status)` of all the lines found in
        the Cobertura report for the given file `filename` where `lineno` is
        the line number and `status` is coverage status of the line which can
        be either `True` (line hit) or `False` (line miss).
        """
        line_elements = self._get_lines_by_filename(filename)

        lines_w_status = []
        for line in line_elements:
            lineno = int(line.attrib['number'])
            status = line.attrib['hits'] != '0'
            lines_w_status.append((lineno, status))

        return lines_w_status
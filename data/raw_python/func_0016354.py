def missed_statements(self, filename):
        """
        Return a list of uncovered line numbers for each of the missed
        statements found for the file `filename`.
        """
        el = self._get_class_element_by_filename(filename)
        lines = el.xpath('./lines/line[@hits=0]')
        return [int(l.attrib['number']) for l in lines]
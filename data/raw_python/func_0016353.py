def branch_rate(self, filename=None):
        """
        Return the global branch rate of the coverage report. If the
        `filename` file is given, return the branch rate of the file.
        """
        if filename is None:
            el = self.xml
        else:
            el = self._get_class_element_by_filename(filename)

        return float(el.attrib['branch-rate'])
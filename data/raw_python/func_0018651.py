def _get_variable_from_filepath(self):
        """
        Determine the file variable from the filepath.

        Returns
        -------
        str
            Best guess of variable name from the filepath
        """
        try:
            return self.regexp_capture_variable.search(self.filepath).group(1)
        except AttributeError:
            self._raise_cannot_determine_variable_from_filepath_error()
def _input_as_paths(self, data):
        """ Return data as a space delimited string with each path quoted

            data: paths or filenames, most likely as a list of
             strings

        """
        return self._command_delimiter.join(
            map(str, map(self._input_as_path, data)))
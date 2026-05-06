def _handle_error(self, text, ErrorClass, infile, cur_index):
        """
        Handle an error according to the error settings.

        Either raise the error or store it.
        The error will have occured at ``cur_index``
        """
        line = infile[cur_index]
        cur_index += 1
        message = text % cur_index
        error = ErrorClass(message, cur_index, line)
        if self.raise_errors:
            # raise the error - parsing stops here
            raise error
        # store the error
        # reraise when parsing has finished
        self._errors.append(error)
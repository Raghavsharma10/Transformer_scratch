def read_bytes(self, count):
        """Read a given number of bytes from the input stream.

        Throws MissingBytes if the bytes are not found.

        Note: This method does not read from the line buffer.

        :return: a string
        """
        result = self.input.read(count)
        found = len(result)
        self.lineno += result.count(b'\n')
        if found != count:
            self.abort(errors.MissingBytes, count, found)
        return result
def _read_multiline(self, init_data):
        """Reads multiline symbols (ususally comments)

        :param init_data: initial data (parsed from the line containing keyword)
        :return: parsed value of the multiline symbol
        :rtype: str
        """
        result = init_data

        first = True
        while True:
            last_index = self.current_file.tell()
            line_raw = self.current_file.readline()

            # don't add newlines from full line comments
            if line_raw[0] == '#':
                continue

            # now strip comments
            # TODO - is it appropriate behavior?
            data = self._strip_comments(line_raw)
            # EOF, stop here
            if not data:
                break

            # we arrived to the next command, step back and break
            if len(data.strip()) >= 1 and data.strip()[0] == '*':
                self.current_file.seek(last_index)
                break

            if first:
                result += '\n'
                first = False

            result += data

        result = result.strip()
        if result and not result.endswith('.'):
            result += '.'

        return result
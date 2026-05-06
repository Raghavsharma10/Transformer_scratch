def _parse_line(self, file):
        """Parse single line (or more if particular keyword actually demands it)

        :param file:
        :type file: file
        """

        line = self._strip_comments(file.readline())
        # check if the file ended
        if not line:
            return False

        # line was empty or it was a comment, continue
        if line.strip() == '':
            return True

        # retrieve keyword and its value
        reg = re.search("^\*(?P<key>[^:#]*)(:\s*(?P<value>.*)\s*)?$", line)
        if reg:
            key = reg.group('key').strip()
            value = reg.group('value')

            if key in self.mapping[self.current_state]:
                self.mapping[self.current_state][key](value)
            elif self.strict:
                raise BadFormat("unknown key: *%s" % key)
        else:
            raise BadFormat("line can not be parsed: %s" % line)

        return True
def _strip_comments(line):
        """Processes line stripping any comments from it

        :param line: line to be processed
        :type line: str
        :return: line with removed comments
        :rtype: str
        """
        if line == '':
            return line
        r = re.search('(?P<line>[^#]*)(#(?P<comment>.*))?', line)
        if r:
            line = r.group('line')
            if not line.endswith('\n'):
                line += '\n'
            return line
        return '\n'
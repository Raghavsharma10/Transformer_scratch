def debug_print_line(self, i, level, line):
        """
        Debug print of the currently parsed line
        :param i: The line number of the line that is being currently parsed
        :param level: Parser level
        :param line: the line that is currently being parsed
        :return: None
        """
        if self.debug_level == 2:
            print("Line %d (%d): '%s'" % (i + 1, level, line.rstrip(' \r\n\t\f')))
        if self.debug_level > 2:
            print("Line %d (%d):" % (i + 1, level))
            hexdump(line)
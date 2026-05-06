def debug_print_strip_msg(self, i, line):
        """
        Debug print indicating that an empty line is being skipped
        :param i: The line number of the line that is being currently parsed
        :param line: the parsed line
        :return: None
        """
        if self.debug_level == 2:
            print("     Stripping Line %d: '%s'" % (i + 1, line.rstrip(' \r\n\t\f')))
        elif self.debug_level > 2:
            print("     Stripping Line %d:" % (i + 1))
            hexdump(line)
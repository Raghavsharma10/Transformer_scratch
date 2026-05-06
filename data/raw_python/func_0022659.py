def strip_empty_lines_forward(self, content, i):
        """
        Skip over empty lines
        :param content: parsed text
        :param i: current parsed line
        :return: number of skipped lined
        """
        while i < len(content):
            line = content[i].strip(' \r\n\t\f')
            if line != '':
                break
            self.debug_print_strip_msg(i, content[i])
            i += 1  # Strip an empty line
        return i
def indent(self, message):
        """
        Sets the indent for standardized output
        :param message: (str)
        :return: (str)
        """
        indent = self.indent_char * self.indent_size
        return indent + message
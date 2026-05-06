def writeln(self, data):
        """
        Write a line of text to the file

        :param data: The text to write
        """
        self.f.write(" "*self.indent_level)
        self.f.write(data + "\n")
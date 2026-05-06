def write_file(self, filename, content):
        """Write a file.

        This is useful when writing a file that will fit within memory

        :param filename: ``str``
        :param content: ``str``
        """
        with open(filename, 'wb') as f:
            self.log.debug(content)
            f.write(content)
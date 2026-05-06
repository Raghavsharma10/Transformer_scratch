def write_file_lines(self, filename, contents):
        """Write a file.

        This is useful when writing a file that may not fit within memory.

        :param filename: ``str``
        :param contents: ``list``
        """
        with open(filename, 'wb') as f:
            self.log.debug(contents)
            f.writelines(contents)
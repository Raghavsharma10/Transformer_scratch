def write(self, file_name):
        """
        Writes the chapter object to an xhtml file.

        Args:
            file_name (str): The full name of the xhtml file to save to.
        """
        try:
            assert file_name[-6:] == '.xhtml'
        except (AssertionError, IndexError):
            raise ValueError('filename must end with .xhtml')
        with open(file_name, 'wb') as f:
            f.write(self.content.encode('utf-8'))
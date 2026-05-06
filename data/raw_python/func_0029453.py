def write(self, use_template=False, out_file_or_path=None, encoding=DEFAULT_ENCODING):
        """
        Validates instance properties, updates an XML tree with them, and writes the content to a file.
        :param use_template: if True, updates a new template XML tree; otherwise the original XML tree
        :param out_file_or_path: optionally override self.out_file_or_path with a custom file path
        :param encoding: optionally use another encoding instead of UTF-8
        """

        if not out_file_or_path:
            out_file_or_path = self.out_file_or_path

        if not out_file_or_path:
            # FileNotFoundError doesn't exist in Python 2
            raise IOError('Output file path has not been provided')

        write_element(self.update(use_template), out_file_or_path, encoding)
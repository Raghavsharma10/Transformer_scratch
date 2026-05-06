def open(self, filepath):
        """
        Open settings backend to return its content

        Args:
            filepath (str): Settings object, depends from backend

        Returns:
            string: File content.

        """
        with io.open(filepath, 'r', encoding='utf-8') as fp:
            content = fp.read()
        return content
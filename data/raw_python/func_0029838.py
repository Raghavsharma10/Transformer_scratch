def default(self):
        """Return default contents"""

        import ambry.bundle.default_files as df
        import os

        path = os.path.join(os.path.dirname(df.__file__), self.file_name)

        if six.PY2:
            with open(path, 'rb') as f:
                return f.read()
        else:
            # py3
            with open(path, 'rt', encoding='utf-8') as f:
                return f.read()
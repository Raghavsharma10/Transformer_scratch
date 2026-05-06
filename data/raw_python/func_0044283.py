def dump(self, content, filepath, indent=4):
        """
        Dump settings content to filepath.

        Args:
            content (str): Settings content.
            filepath (str): Settings file location.
        """
        with open(filepath, 'w') as fp:
            json.dump(content, fp, indent=indent)
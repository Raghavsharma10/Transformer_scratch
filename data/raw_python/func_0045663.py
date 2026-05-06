def write(self, path=None, *args, **kwargs):
        """
        Perform formatting and write the formatted string to a file or stdout.

        Optional arguments can be used to format the editor's contents. If no
        file path is given, prints to standard output.

        Args:
            path (str): Full file path (default None, prints to stdout)
            *args: Positional arguments to format the editor with
            **kwargs: Keyword arguments to format the editor with
        """
        if path is None:
            print(self.format(*args, **kwargs))
        else:
            with io.open(path, 'w', newline="") as f:
                f.write(self.format(*args, **kwargs))
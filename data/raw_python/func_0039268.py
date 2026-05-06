def output(self):
        """
        Create full path for excel file to save parsed translations strings.
        Returns:
            unicode: full path for excel file to save parsed translations strings.
        """

        path, src = os.path.split(self.src)
        src, ext = os.path.splitext(src)

        return os.path.join(path, "{src}.xls".format(**{"src": src, }))
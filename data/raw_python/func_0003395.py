def saveto(self, path, sortkey = True):
        """
        Save configurations to path
        """
        with open(path, 'w') as f:
            self.savetofile(f, sortkey)
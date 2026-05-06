def imprint(self, path=None):
        """Write the determined version, if any, to ``self.version_file`` or
           the path passed as an argument.
        """
        if self.version is not None:
            with open(path or self.version_file, 'w') as h:
                h.write(self.version + '\n')
        else:
            raise ValueError('Can not write null version to file.')
        return self
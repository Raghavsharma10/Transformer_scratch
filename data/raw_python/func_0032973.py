def output(self):
        """ Use the digest version, since URL can be ugly. """
        return luigi.LocalTarget(path=self.path(digest=True, ext='html'))
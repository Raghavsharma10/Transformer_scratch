def _absolute(self, path):
        """ Convert a filename to an absolute path """
        path = FilePath(path)
        if isabs(path):
            return path
        else:
            # these are both Path objects, so joining with + is acceptable
            return self.WorkingDir + path
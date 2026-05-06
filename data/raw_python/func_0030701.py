def globChildren(self, pattern):
        """
        Assuming I am representing a directory, return a list of
        FilePaths representing my children that match the given
        pattern.
        """
        import glob
        path = self.path[-1] == '/' and self.path + pattern or slash.join([self.path, pattern])
        return map(self.clonePath, glob.glob(path))
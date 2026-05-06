def preauthChild(self, path):
        """
        Use me if `path' might have slashes in it, but you know they're safe.

        (NOT slashes at the beginning. It still needs to be a _child_).
        """
        newpath = abspath(joinpath(self.path, normpath(path)))
        if not newpath.startswith(self.path):
            raise InsecurePath("%s is not a child of %s" % (newpath, self.path))
        return self.clonePath(newpath)
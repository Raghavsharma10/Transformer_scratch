def ext(self, extension):
        """
        Match files with an extension - e.g. 'js', 'txt'
        """
        new_pathq = copy(self)
        new_pathq._pattern.ext = extension
        return new_pathq
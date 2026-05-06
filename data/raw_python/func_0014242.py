def match(self, fname, flevel, ftype):
        '''Returns the result score if the file matches this rule'''
        # if filetype is the same
        # and level isn't set or level is the same
        # and pattern matche the filename
        if self.filetype == ftype and (self.level is None or self.level == flevel) and fnmatch.fnmatch(fname, self.pattern):
            return self.score
        return 0
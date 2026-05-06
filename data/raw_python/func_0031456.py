def read_files(self, condition='*'):
        """Read specific files from archive into memory.
        If "condition" is a list of numbers, then return files which have those positions in infolist.
        If "condition" is a string, then it is treated as a wildcard for names of files to extract.
        If "condition" is a function, it is treated as a callback function, which accepts a RarInfo object 
            and returns boolean True (extract) or False (skip).
        If "condition" is omitted, all files are returned.
        
        Returns list of tuples (RarInfo info, str contents)
        """
        checker = condition2checker(condition)
        return RarFileImplementation.read_files(self, checker)
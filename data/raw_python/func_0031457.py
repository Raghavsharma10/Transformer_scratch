def extract(self, condition='*', path='.', withSubpath=True,
                overwrite=True):
        """Extract specific files from archive to disk.
        
        If "condition" is a list of numbers, then extract files which have those positions in infolist.
        If "condition" is a string, then it is treated as a wildcard for names of files to extract.
        If "condition" is a function, it is treated as a callback function, which accepts a RarInfo object
            and returns either boolean True (extract) or boolean False (skip).
        DEPRECATED: If "condition" callback returns string (only supported for Windows) - 
            that string will be used as a new name to save the file under.
        If "condition" is omitted, all files are extracted.
        
        "path" is a directory to extract to
        "withSubpath" flag denotes whether files are extracted with their full path in the archive.
        "overwrite" flag denotes whether extracted files will overwrite old ones. Defaults to true.
        
        Returns list of RarInfos for extracted files."""
        checker = condition2checker(condition)
        return RarFileImplementation.extract(self, checker, path, withSubpath,
                                             overwrite)
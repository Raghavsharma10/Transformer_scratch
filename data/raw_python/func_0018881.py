def filepaths(self) -> List[str]:
        """Absolute path names of the files contained in the current
        working directory.

        Files names starting with underscores are ignored:

        >>> from hydpy.core.filetools import FileManager
        >>> filemanager = FileManager()
        >>> filemanager.BASEDIR = 'basename'
        >>> filemanager.projectdir = 'projectname'
        >>> from hydpy import repr_, TestIO
        >>> with TestIO():
        ...     filemanager.currentdir = 'testdir'
        ...     open('projectname/basename/testdir/file1.txt', 'w').close()
        ...     open('projectname/basename/testdir/file2.npy', 'w').close()
        ...     open('projectname/basename/testdir/_file1.nc', 'w').close()
        ...     for filepath in filemanager.filepaths:
        ...         repr_(filepath)    # doctest: +ELLIPSIS
        '...hydpy/tests/iotesting/projectname/basename/testdir/file1.txt'
        '...hydpy/tests/iotesting/projectname/basename/testdir/file2.npy'
        """
        path = self.currentpath
        return [os.path.join(path, name) for name in self.filenames]
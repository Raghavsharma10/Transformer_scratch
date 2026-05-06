def filenames(self) -> List[str]:
        """Names of the files contained in the the current working directory.

        Files names starting with underscores are ignored:

        >>> from hydpy.core.filetools import FileManager
        >>> filemanager = FileManager()
        >>> filemanager.BASEDIR = 'basename'
        >>> filemanager.projectdir = 'projectname'
        >>> from hydpy import TestIO
        >>> with TestIO():
        ...     filemanager.currentdir = 'testdir'
        ...     open('projectname/basename/testdir/file1.txt', 'w').close()
        ...     open('projectname/basename/testdir/file2.npy', 'w').close()
        ...     open('projectname/basename/testdir/_file1.nc', 'w').close()
        ...     filemanager.filenames
        ['file1.txt', 'file2.npy']
        """
        return sorted(
            fn for fn in os.listdir(self.currentpath)
            if not fn.startswith('_'))
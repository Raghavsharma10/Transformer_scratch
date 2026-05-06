def currentpath(self) -> str:
        """Absolute path of the current working directory.

        >>> from hydpy.core.filetools import FileManager
        >>> filemanager = FileManager()
        >>> filemanager.BASEDIR = 'basename'
        >>> filemanager.projectdir = 'projectname'
        >>> from hydpy import repr_, TestIO
        >>> with TestIO():
        ...     filemanager.currentdir = 'testdir'
        ...     repr_(filemanager.currentpath)    # doctest: +ELLIPSIS
        '...hydpy/tests/iotesting/projectname/basename/testdir'
        """
        return os.path.join(self.basepath, self.currentdir)
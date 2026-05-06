def availabledirs(self) -> Folder2Path:
        """Names and paths of the available working directories.

        Available working directories are those beeing stored in the
        base directory of the respective |FileManager| subclass.
        Folders with names starting with an underscore are ignored
        (use this for directories handling additional data files,
        if you like).  Zipped directories, which can be unpacked
        on the fly, do also count as available directories:

        >>> from hydpy.core.filetools import FileManager
        >>> filemanager = FileManager()
        >>> filemanager.BASEDIR = 'basename'
        >>> filemanager.projectdir = 'projectname'
        >>> import os
        >>> from hydpy import repr_, TestIO
        >>> TestIO.clear()
        >>> with TestIO():
        ...     os.makedirs('projectname/basename/folder1')
        ...     os.makedirs('projectname/basename/folder2')
        ...     open('projectname/basename/folder3.zip', 'w').close()
        ...     os.makedirs('projectname/basename/_folder4')
        ...     open('projectname/basename/folder5.tar', 'w').close()
        ...     filemanager.availabledirs   # doctest: +ELLIPSIS
        Folder2Path(folder1=.../projectname/basename/folder1,
                    folder2=.../projectname/basename/folder2,
                    folder3=.../projectname/basename/folder3.zip)
        """
        directories = Folder2Path()
        for directory in os.listdir(self.basepath):
            if not directory.startswith('_'):
                path = os.path.join(self.basepath, directory)
                if os.path.isdir(path):
                    directories.add(directory, path)
                elif directory.endswith('.zip'):
                    directories.add(directory[:-4], path)
        return directories
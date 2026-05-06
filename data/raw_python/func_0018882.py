def zip_currentdir(self) -> None:
        """Pack the current working directory in a `zip` file.

        |FileManager| subclasses allow for manual packing and automatic
        unpacking of working directories.  The only supported format is `zip`.
        To avoid possible inconsistencies, origin directories and zip
        files are removed after packing or unpacking, respectively.

        As an example scenario, we prepare a |FileManager| object with
        the current working directory `folder` containing the files
        `test1.txt` and `text2.txt`:

        >>> from hydpy.core.filetools import FileManager
        >>> filemanager = FileManager()
        >>> filemanager.BASEDIR = 'basename'
        >>> filemanager.projectdir = 'projectname'
        >>> import os
        >>> from hydpy import repr_, TestIO
        >>> TestIO.clear()
        >>> basepath = 'projectname/basename'
        >>> with TestIO():
        ...     os.makedirs(basepath)
        ...     filemanager.currentdir = 'folder'
        ...     open(f'{basepath}/folder/file1.txt', 'w').close()
        ...     open(f'{basepath}/folder/file2.txt', 'w').close()
        ...     filemanager.filenames
        ['file1.txt', 'file2.txt']

        The directories existing under the base path are identical
        with the ones returned by property |FileManager.availabledirs|:

        >>> with TestIO():
        ...     sorted(os.listdir(basepath))
        ...     filemanager.availabledirs    # doctest: +ELLIPSIS
        ['folder']
        Folder2Path(folder=.../projectname/basename/folder)

        After packing the current working directory manually, it is
        still counted as a available directory:

        >>> with TestIO():
        ...     filemanager.zip_currentdir()
        ...     sorted(os.listdir(basepath))
        ...     filemanager.availabledirs    # doctest: +ELLIPSIS
        ['folder.zip']
        Folder2Path(folder=.../projectname/basename/folder.zip)

        Instead of the complete directory, only the contained files
        are packed:

        >>> from zipfile import ZipFile
        >>> with TestIO():
        ...     with ZipFile('projectname/basename/folder.zip', 'r') as zp:
        ...         sorted(zp.namelist())
        ['file1.txt', 'file2.txt']

        The zip file is unpacked again, as soon as `folder` becomes
        the current working directory:

        >>> with TestIO():
        ...     filemanager.currentdir = 'folder'
        ...     sorted(os.listdir(basepath))
        ...     filemanager.availabledirs
        ...     filemanager.filenames    # doctest: +ELLIPSIS
        ['folder']
        Folder2Path(folder=.../projectname/basename/folder)
        ['file1.txt', 'file2.txt']
        """
        with zipfile.ZipFile(f'{self.currentpath}.zip', 'w') as zipfile_:
            for filepath, filename in zip(self.filepaths, self.filenames):
                zipfile_.write(filename=filepath, arcname=filename)
        del self.currentdir
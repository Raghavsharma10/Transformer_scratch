def currentdir(self) -> str:
        """Name of the current working directory containing the relevant files.

        To show most of the functionality of |property|
        |FileManager.currentdir| (unpacking zip files on the fly is
        explained in the documentation on function
        (|FileManager.zip_currentdir|), we first prepare a |FileManager|
        object corresponding to the |FileManager.basepath|
        `projectname/basename`:

        >>> from hydpy.core.filetools import FileManager
        >>> filemanager = FileManager()
        >>> filemanager.BASEDIR = 'basename'
        >>> filemanager.projectdir = 'projectname'
        >>> import os
        >>> from hydpy import repr_, TestIO
        >>> TestIO.clear()
        >>> with TestIO():
        ...     os.makedirs('projectname/basename')
        ...     repr_(filemanager.basepath)    # doctest: +ELLIPSIS
        '...hydpy/tests/iotesting/projectname/basename'

        At first, the base directory is empty and asking for the
        current working directory results in the following error:

        >>> with TestIO():
        ...     filemanager.currentdir   # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        RuntimeError: The current working directory of the FileManager object \
has not been defined manually and cannot be determined automatically: \
`.../projectname/basename` does not contain any available directories.

        If only one directory exists, it is considered as the current
        working directory automatically:

        >>> with TestIO():
        ...     os.mkdir('projectname/basename/dir1')
        ...     filemanager.currentdir
        'dir1'

        |property| |FileManager.currentdir| memorises the name of the
        current working directory, even if another directory is later
        added to the base path:

        >>> with TestIO():
        ...     os.mkdir('projectname/basename/dir2')
        ...     filemanager.currentdir
        'dir1'

        Set the value of |FileManager.currentdir| to |None| to let it
        forget the memorised directory.  After that, asking for the
        current working directory now results in another error, as
        it is not clear which directory to select:

        >>> with TestIO():
        ...     filemanager.currentdir = None
        ...     filemanager.currentdir   # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        RuntimeError: The current working directory of the FileManager object \
has not been defined manually and cannot be determined automatically: \
`....../projectname/basename` does contain multiple available directories \
(dir1 and dir2).

        Setting |FileManager.currentdir| manually solves the problem:

        >>> with TestIO():
        ...     filemanager.currentdir = 'dir1'
        ...     filemanager.currentdir
        'dir1'

        Remove the current working directory `dir1` with the `del` statement:

        >>> with TestIO():
        ...     del filemanager.currentdir
        ...     os.path.exists('projectname/basename/dir1')
        False

        |FileManager| subclasses can define a default directory name.
        When many directories exist and none is selected manually, the
        default directory is selected automatically.  The following
        example shows an error message due to multiple directories
        without any having the default name:

        >>> with TestIO():
        ...     os.mkdir('projectname/basename/dir1')
        ...     filemanager.DEFAULTDIR = 'dir3'
        ...     del filemanager.currentdir
        ...     filemanager.currentdir   # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        RuntimeError: The current working directory of the FileManager object \
has not been defined manually and cannot be determined automatically: The \
default directory (dir3) is not among the available directories (dir1 and dir2).

        We can fix this by adding the required default directory manually:

        >>> with TestIO():
        ...     os.mkdir('projectname/basename/dir3')
        ...     filemanager.currentdir
        'dir3'

        Setting the |FileManager.currentdir| to `dir4` not only overwrites
        the default name, but also creates the required folder:

        >>> with TestIO():
        ...     filemanager.currentdir = 'dir4'
        ...     filemanager.currentdir
        'dir4'
        >>> with TestIO():
        ...     sorted(os.listdir('projectname/basename'))
        ['dir1', 'dir2', 'dir3', 'dir4']

        Failed attempts in removing directories result in error messages
        like the following one:

        >>> import shutil
        >>> from unittest.mock import patch
        >>> with patch.object(shutil, 'rmtree', side_effect=AttributeError):
        ...     with TestIO():
        ...         del filemanager.currentdir   # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        AttributeError: While trying to delete the current working directory \
`.../projectname/basename/dir4` of the FileManager object, the following \
error occurred: ...

        Then, the current working directory still exists and is remembered
        by |FileManager.currentdir|:

        >>> with TestIO():
        ...     filemanager.currentdir
        'dir4'
        >>> with TestIO():
        ...     sorted(os.listdir('projectname/basename'))
        ['dir1', 'dir2', 'dir3', 'dir4']
        """
        if self._currentdir is None:
            directories = self.availabledirs.folders
            if len(directories) == 1:
                self.currentdir = directories[0]
            elif self.DEFAULTDIR in directories:
                self.currentdir = self.DEFAULTDIR
            else:
                prefix = (f'The current working directory of the '
                          f'{objecttools.classname(self)} object '
                          f'has not been defined manually and cannot '
                          f'be determined automatically:')
                if not directories:
                    raise RuntimeError(
                        f'{prefix} `{objecttools.repr_(self.basepath)}` '
                        f'does not contain any available directories.')
                if self.DEFAULTDIR is None:
                    raise RuntimeError(
                        f'{prefix} `{objecttools.repr_(self.basepath)}` '
                        f'does contain multiple available directories '
                        f'({objecttools.enumeration(directories)}).')
                raise RuntimeError(
                    f'{prefix} The default directory ({self.DEFAULTDIR}) '
                    f'is not among the available directories '
                    f'({objecttools.enumeration(directories)}).')
        return self._currentdir
def showpath(path):
        """Return path in form most convenient for user to read.

        Return relative path when input path is within the current working
        directory, otherwise return same (absolute) path passed in.

        :param path: file system path
        :type path: str or unicode
        :returns: file system path
        :rtype: str

        """
        try:
            retval = os.path.relpath(path, os.getcwd())
        except ValueError:
            retval = path
        else:
            if retval.startswith('..'):
                retval = path
        return retval
def getcwd(cls):
        """
        Provide a context dependent current working directory. This method
        will return the directory currently holding the lock.
        """
        if not hasattr(cls._tl, "cwd"):
            cls._tl.cwd = os.getcwd()
        return cls._tl.cwd
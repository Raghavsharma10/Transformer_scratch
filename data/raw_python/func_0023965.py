def lock(self):
        '''
        Try to get locked the file
        - the function will wait until the file is unlocked if 'wait' was defined as locktype
        - the funciton will raise AlreadyLocked exception if 'lock' was defined as locktype
        '''

        # Open file
        self.__fd = open(self.__lockfile, "w")

        # Get it locked
        if self.__locktype == "wait":
            # Try to get it locked until ready
            fcntl.flock(self.__fd.fileno(), fcntl.LOCK_EX)
        elif self.__locktype == "lock":
            # Try to get the locker if can not raise an exception
            try:
                fcntl.flock(self.__fd.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
            except IOError:
                raise AlreadyLocked("File is already locked")
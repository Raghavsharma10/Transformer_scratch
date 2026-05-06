def __checkExpiration(self, mtime=None):
        '''
            __checkExpiration - Check if we have expired
            
            @param mtime <int> - Optional mtime if known, otherwise will be gathered

            @return <bool> - True if we did expire, otherwise False
        '''
        if not self.maxLockAge:
            return False

        if mtime is None:
            try:
                mtime = os.stat(self.lockPath).st_mtime
            except FileNotFoundError as e:
                return False

        if mtime < time.time() - self.maxLockAge:
            return True

        return False
def isHeld(self):
        '''
            isHeld - True if anyone holds the lock, otherwise False.

            @return bool - If lock is held by anyone
        '''
        if not os.path.exists(self.lockPath):
            return False
        
        try:
            mtime = os.stat(self.lockPath).st_mtime
        except FileNotFoundError as e:
            return False
        
        if self.__checkExpiration(mtime):
            return False

        return True
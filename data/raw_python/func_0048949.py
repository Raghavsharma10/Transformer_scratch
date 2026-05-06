def release(self, forceRelease=False):
        '''
            release - Release the lock.

            @param forceRelease <bool> default False - If True, will release the lock even if we don't hold it.

            @return - True if lock is released, otherwise False
        '''
        if not self.held:
            if forceRelease is False:
                return False # We were not holding the lock
            else:
                self.held = True # If we have force release set, pretend like we held its
        
        if not os.path.exists(self.lockPath):
            self.held = False
            self.acquiredAt = None
            return True

        if forceRelease is False:
            # We waited too long and lost the lock
            if self.maxLockAge and time.time() > self.acquiredAt + self.maxLockAge:
                self.held = False
                self.acquiredAt = None
                return False

        self.acquiredAt = None

        try:
            os.rmdir(self.lockPath)
            self.held = False
            return True
        except:
            self.held = False
            return False
def hasLock(self):
        '''
            hasLock - Property, returns True if we have the lock, or False if we do not.

            @return <bool> - True/False if we have the lock or not.
        '''
        # If we don't hold it currently, return False
        if self.held is False:
            return False
        
        # Otherwise if we think we hold it, but it is not held, we have lost it.
        if not self.isHeld:
            self.acquiredAt = None
            self.held = False
            return False

        # Check if we expired
        if self.__checkExpiration(self.acquiredAt):
            self.acquiredAt = None
            self.held = False
            return False


        return True
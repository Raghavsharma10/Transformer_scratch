def _lock(self):
        '''
        Locks, or returns False if already locked
        '''
        if not self._is_locked():
            with open(self._lck,'w') as fh:
                if self._devel: self.logger.debug("Locking")
                fh.write(str(os.getpid()))
            return True
        else:
            return False
def _unlock(self):
        '''
        Unlocks the index
        '''
        if self._devel: self.logger.debug("Unlocking Index")
        if self._is_locked():
            os.remove(self._lck)
            return True
        else:
            return True
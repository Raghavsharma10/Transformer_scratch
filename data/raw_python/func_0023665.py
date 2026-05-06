def delay(self):
        '''How long to wait before the next check'''
        if self._last_checked:
            return self._interval - (time.time() - self._last_checked)
        return self._interval
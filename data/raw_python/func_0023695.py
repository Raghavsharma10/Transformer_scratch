def ready(self):
        '''Whether or not enough time has passed since the last failure'''
        if self._last_failed:
            delta = time.time() - self._last_failed
            return delta >= self.backoff()
        return True
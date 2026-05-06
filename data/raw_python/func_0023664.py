def wait(self, timeout):
        '''Wait for the provided time to elapse'''
        logger.debug('Waiting for %fs', timeout)
        return self._event.wait(timeout)
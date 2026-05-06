def callback(self):
        '''Run the callback'''
        self._callback(*self._args, **self._kwargs)
        self._last_checked = time.time()
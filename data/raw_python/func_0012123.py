def on_finish(self, func):
        '''Assign a callback function to be run when successfully complete

        :param function func:
            A callback to run when complete. It will be given one argument (the
            value that has arrived), and it's return value is ignored.
        '''
        if self._done.is_set():
            if self._failure is None:
                backend.schedule(func, args=(self._value,))
        else:
            self._cbacks.append(func)
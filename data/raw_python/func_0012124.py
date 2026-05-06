def on_abort(self, func):
        '''Assign a callback function to be run when :meth:`abort`\ed

        :param function func:
            A callback to run when aborted. It will be given three arguments:

                - ``klass``: the exception class
                - ``exc``: the exception instance
                - ``tb``: the traceback object associated with the exception
        '''
        if self._done.is_set():
            if self._failure is not None:
                backend.schedule(func, args=self._failure)
        else:
            self._errbacks.append(func)
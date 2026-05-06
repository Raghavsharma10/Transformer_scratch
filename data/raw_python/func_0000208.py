def run(self):
        """Modified ``run`` that captures return value and exceptions from ``target``"""
        try:
            if self._target:
                return_value = self._target(*self._args, **self._kwargs)
                if return_value is not None:
                    self._exception = OrphanedReturn(self, return_value)
        except BaseException as err:
            self._exception = err
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs
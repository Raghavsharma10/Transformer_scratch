def handle_cancel(self, **kwargs):
        '''Cancelling a run.
        '''
        for func in self._cancel_functions:
            f_args = getargspec(func)[0]
            f_kwargs = {key: kwargs[key] for key in f_args if key in kwargs}
            func(**f_kwargs)
def print(self, *args, **kwargs):
        '''
        Utility function that behaves identically to 'print' except it only
        prints if verbose
        '''
        if self._last_args and self._last_args.verbose:
            print(*args, **kwargs)
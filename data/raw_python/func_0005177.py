def time(self, intervals=1, *args, _show_progress=True, _print=True,
             _collect_garbage=True, _quiet=True, **kwargs):
        """ Measures the execution time of :prop:_callable for @intervals

            @intervals: #int number of intervals to measure the execution time
                of the function for
            @*args: arguments to pass to the callable being timed
            @**kwargs: arguments to pass to the callable being timed
            @_show_progress: #bool whether or not to print a progress bar
            @_print: #bool whether or not to print the results of the timing
            @_collect_garbage: #bool whether or not to garbage collect
                while timing
            @_quiet: #bool whether or not to disable the print() function's
                ability to output to terminal during the timing

            -> :class:collections.OrderedDict of stats about the timing
        """
        self.reset()
        args = list(args) + list(self._callableargs[0])
        _kwargs = self._callableargs[1]
        _kwargs.update(kwargs)
        kwargs = _kwargs
        if not _collect_garbage:
            gc.disable()  # Garbage collection setting
        gc.collect()
        self.allocated_memory = 0
        for x in self.progress(intervals):
            if _quiet:  # Quiets print()s in the tested function
                sys.stdout = NullIO()
            try:
                self.start()  # Starts the timer
                self._callable(*args, **kwargs)
                self.stop()  # Stops the timer
            except Exception as e:
                if _quiet:  # Unquiets prints()
                    sys.stdout = sys.__stdout__
                raise e
            if _quiet:  # Unquiets prints()
                sys.stdout = sys.__stdout__
        if not _collect_garbage:
            gc.enable()  # Garbage collection setting
        if _print:
            self.info()
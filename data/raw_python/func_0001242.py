def _start_callables(self, row, callables):
        """Start running `callables` asynchronously.
        """
        id_vals = {c: row[c] for c in self.ids}

        def callback(tab, cols, result):
            if isinstance(result, Mapping):
                pass
            elif isinstance(result, tuple):
                result = dict(zip(cols, result))
            elif len(cols) == 1:
                # Don't bother raising an exception if cols != 1
                # because it would be lost in the thread.
                result = {cols[0]: result}
            result.update(id_vals)
            tab._write(result)

        if self._pool is None:
            self._pool = Pool()
        if self._lock is None:
            self._lock = multiprocessing.Lock()

        for cols, fn in callables:
            cb_func = partial(callback, self, cols)

            gen = None
            if inspect.isgeneratorfunction(fn):
                gen = fn()
            elif inspect.isgenerator(fn):
                gen = fn

            if gen:
                def callback_for_each():
                    for i in gen:
                        cb_func(i)
                self._pool.apply_async(callback_for_each)
            else:
                self._pool.apply_async(fn, callback=cb_func)
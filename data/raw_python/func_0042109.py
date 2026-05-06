def call(self, **kwargs):
        """Call all the functions that have previously been added to the
        dependency graph in topological and lexicographical order, and
        then return variables in a ``dict``.

        You may provide variable values with keyword arguments.  These
        values will be written and can satisfy dependencies.

        NOTE: This object will be **destroyed** after ``call()`` returns
        and should not be used any further.
        """
        if not hasattr(self, 'funcs'):
            raise StartupError('startup cannot be called again')
        for name, var in self.variables.items():
            var.name = name
        self.variable_values.update(kwargs)
        for name in self.variable_values:
            self.variables[name].name = name
        queue = Closure.sort(self.satisfied)
        queue.extend(_write_values(self.variable_values, self.variables))
        while queue:
            closure = queue.pop(0)
            writeto = closure.call()
            self.funcs.remove(closure.func)
            queue.extend(_notify_reader_writes(writeto))
        if self.funcs:
            raise StartupError('cannot satisfy dependency for %r' % self.funcs)
        values = {
            name: var.read_latest() for name, var in self.variables.items()
        }
        # Call _release() on normal exit only; otherwise keep the dead body for
        # forensic analysis.
        self._release()
        return values
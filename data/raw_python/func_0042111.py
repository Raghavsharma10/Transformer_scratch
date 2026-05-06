def call(self):
        """Call the closure and return variable(s) that is written."""
        assert self.satisfied, self
        LOG.debug('call %s.%s', self.func.__module__, self.func.__qualname__)
        kwargs = {arg.name: arg.read() for arg in self.args}
        out_value = self.func(**kwargs)
        if self.writeto is None:
            writeto = set()
        elif isinstance(self.writeto, Variable):
            self.writeto.write(out_value)
            writeto = {self.writeto}
        else:
            # A variable can be written multiple times, but we only
            # return a unique set of variables.
            for var, value in zip(self.writeto, out_value):
                var.write(value)
            writeto = set(self.writeto)
        self._release()  # Only call _release() on normal exit.
        return writeto
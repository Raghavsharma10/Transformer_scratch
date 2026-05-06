def get(self, name=None):
        """Get initial yield value, or result of send(name) if name given."""
        if name is None:
            return self.init_value
        elif not self.support_name:
            msg = "generator does not support get-by-name: function {!r}"
            raise TypeError(msg.format(self.function))
        try:
            value = self.generator.send(name)
        except StopIteration:
            msg = "generator didn't yield: function {!r}"
            raise RuntimeError(msg.format(self.function))
        return value
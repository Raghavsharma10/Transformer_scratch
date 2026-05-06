def changed(self, message=None, *args):
        """Marks the object as changed.

        If a `parent` attribute is set, the `changed()` method on the parent
        will be called, propagating the change notification up the chain.

        The message (if provided) will be debug logged.
        """
        if message is not None:
            self.logger.debug('%s: %s', self._repr(), message % args)
        self.logger.debug('%s: changed', self._repr())
        if self.parent is not None:
            self.parent.changed()
        elif isinstance(self, Mutable):
            super(TrackedObject, self).changed()
def unblock(self, callback=None):
        """ Unblock this emitter. See :func:`event.EventEmitter.block`.
        
        Note: Use of ``unblock(None)`` only reverses the effect of 
        ``block(None)``; it does not unblock callbacks that were explicitly 
        blocked using ``block(callback)``. 
        """
        if callback not in self._blocked or self._blocked[callback] == 0:
            raise RuntimeError("Cannot unblock %s for callback %s; emitter "
                               "was not previously blocked." % 
                               (self, callback))
        b = self._blocked[callback] - 1
        if b == 0 and callback is not None:
            del self._blocked[callback]
        else:
            self._blocked[callback] = b
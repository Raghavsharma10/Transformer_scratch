def pop(self, name=None):
        """
        Reverts to the nest stage just before the corresponding call of
        :meth:`SConsWrap.add_aggregate`.  However, any aggregate collections
        which have been worked on will still be accessible, and can be called
        operated on together after calling this method.  If no name is passed,
        will revert to the last nest level.

        :param name: Name of the nest level to pop.
        """
        if name is not None:
            self.nest = self.checkpoints[name]
            keys = list(self.checkpoints.keys())
            name_idx = keys.index(name)
            assert name_idx >= 0

            # Pop every key from ``name`` on:
            for k in reversed(keys[name_idx:]):
                self.checkpoints.pop(k)
        else:
            self.nest = self.checkpoints.popitem()[1]
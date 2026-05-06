def process(self, tup):
        """Group non-tick Tuples into batches by ``group_key``.

        .. warning::
            This method should **not** be overriden.  If you want to tweak
            how Tuples are grouped into batches, override ``group_key``.
        """
        # Append latest Tuple to batches
        group_key = self.group_key(tup)
        self._batches[group_key].append(tup)
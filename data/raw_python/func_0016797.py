def prepare(self):
        """Constructs a :class:`~bloop.search.PreparedSearch`."""
        p = PreparedSearch()
        p.prepare(
            engine=self.engine,
            mode=self.mode,
            model=self.model,
            index=self.index,
            key=self.key,
            filter=self.filter,
            projection=self.projection,
            consistent=self.consistent,
            forward=self.forward,
            parallel=self.parallel
        )
        return p
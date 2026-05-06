def build_table(self, table, force=False):
        """Build all of the sources for a table """

        sources = self._resolve_sources(None, [table])

        for source in sources:
            self.build_source(None, source, force=force)

        self.unify_partitions()
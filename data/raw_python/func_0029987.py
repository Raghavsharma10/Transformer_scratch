def source_schema(self, sources=None, tables=None, clean=False):
        """Process a collection of ingested sources to make source tables. """

        sources = self._resolve_sources(sources, tables, None,
                                        predicate=lambda s: s.is_processable)

        for source in sources:
            source.update_table()
            self.log("Creating source schema for '{}': Table {},  {} columns"
                     .format(source.name, source.source_table.name, len(source.source_table.columns)))

        self.commit()
def _resolve_sources(self, sources, tables, stage=None, predicate=None):
        """
        Determine what sources to run from an input of sources and tables

        :param sources:  A collection of source objects, source names, or source vids
        :param tables: A collection of table names
        :param stage: If not None, select only sources from this stage
        :param predicate: If not none, a callable that selects a source to return when True
        :return:
        """

        assert sources is None or tables is None

        if not sources:
            if tables:
                sources = list(s for s in self.sources if s.dest_table_name in tables)
            else:
                sources = self.sources

        elif not isinstance(sources, (list, tuple)):
            sources = [sources]

        def objectify(source):
            if isinstance(source, basestring):
                source_name = source
                return self.source(source_name)
            else:
                return source

        sources = [objectify(s) for s in sources]

        if predicate:
            sources = [s for s in sources if predicate(s)]

        if stage:
            sources = [s for s in sources if str(s.stage) == str(stage)]

        return sources
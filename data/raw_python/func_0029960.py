def source_pipe(self, source, ps=None):
        """Create a source pipe for a source, giving it access to download files to the local cache"""

        if isinstance(source, string_types):
            source = self.source(source)

        source.dataset = self.dataset
        source._bundle = self

        iter_source, source_pipe = self._iterable_source(source, ps)

        if self.limited_run:
            source_pipe.limit = 500

        return iter_source, source_pipe
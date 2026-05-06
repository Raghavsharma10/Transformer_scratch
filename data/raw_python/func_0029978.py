def clean_sources(self):
        """Like clean, but also clears out files. """

        for src in self.dataset.sources:
            src.st_id = None
            src.t_id = None

        self.dataset.sources[:] = []
        self.dataset.source_tables[:] = []
        self.dataset.st_sequence_id = 1
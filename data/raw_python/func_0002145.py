def _get_metadata(self):
        """Get header information and store as metadata for the endpoint."""
        self.metadata = self.fetch_header()
        self.variables = {g.name for g in self.metadata.grids}
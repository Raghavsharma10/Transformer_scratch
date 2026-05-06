def add_self_source(self):
        """Add a source that refers to the catalog itself.

        For now this points to the Open Supernova Catalog by default.
        """
        return self.add_source(
            bibcode=self.catalog.OSC_BIBCODE,
            name=self.catalog.OSC_NAME,
            url=self.catalog.OSC_URL,
            secondary=True)
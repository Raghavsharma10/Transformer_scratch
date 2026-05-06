def sources(self):
        """Iterate over downloadable sources"""

        def set_bundle(s):
            s._bundle = self
            return s
        return list(set_bundle(s) for s in self.dataset.sources)
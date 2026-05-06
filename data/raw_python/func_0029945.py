def refs(self):
        """Iterate over downloadable sources -- references and templates"""

        def set_bundle(s):
            s._bundle = self
            return s

        return list(set_bundle(s) for s in self.dataset.sources if not s.is_downloadable)
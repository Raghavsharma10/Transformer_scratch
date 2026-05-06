def add_photometry(self, compare_to_existing=True, **kwargs):
        """Add a `Photometry` instance to this entry."""
        self._add_cat_dict(
            Photometry,
            self._KEYS.PHOTOMETRY,
            compare_to_existing=compare_to_existing,
            **kwargs)
        return
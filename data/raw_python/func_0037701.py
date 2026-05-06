def evergreen_video(self, **kwargs):
        """Filter evergreen content to exclusively video content."""
        eqs = self.evergreen(**kwargs)
        eqs = eqs.filter(VideohubVideo())
        return eqs
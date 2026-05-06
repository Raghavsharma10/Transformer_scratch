def new_version(self, created_at: dt.datetime, expires_at: dt.datetime=None) -> models.Version:
        """Create a new bundle version."""
        new_version = self.Version(created_at=created_at, expires_at=expires_at)
        return new_version
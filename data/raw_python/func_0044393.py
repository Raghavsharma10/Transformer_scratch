def version(self, bundle: str, date: dt.datetime) -> models.Version:
        """Fetch a version from the store."""
        return (self.Version.query
                            .join(models.Version.bundle)
                            .filter(models.Bundle.name == bundle,
                                    models.Version.created_at == date)
                            .first())
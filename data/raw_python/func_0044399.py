def files(self, *, bundle: str=None, tags: List[str]=None, version: int=None,
              path: str=None) -> models.File:
        """Fetch files from the store."""
        query = self.File.query
        if bundle:
            query = (query.join(self.File.version, self.Version.bundle)
                          .filter(self.Bundle.name == bundle))

        if tags:
            # require records to match ALL tags
            query = (
                query.join(self.File.tags)
                .filter(self.Tag.name.in_(tags))
                .group_by(models.File.id)
                .having(func.count(models.Tag.name) == len(tags))
            )

        if version:
            query = query.join(self.File.version).filter(self.Version.id == version)

        if path:
            query = query.filter_by(path=path)

        return query
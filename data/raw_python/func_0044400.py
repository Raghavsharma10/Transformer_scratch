def files_before(self, *, bundle: str=None, tags: List[str]=None, before:
                     str=None) -> models.File:
        """Fetch files before date from store"""
        query = self.files(tags=tags, bundle=bundle)
        if before:
            before_dt = parse_date(before)
            query = query.join(models.Version).filter(models.Version.created_at < before_dt)

        return query
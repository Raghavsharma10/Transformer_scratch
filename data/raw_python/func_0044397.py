def new_file(self, path: str, checksum: str=None, to_archive: bool=False,
                 tags: List[models.Tag]=None) -> models.File:
        """Create a new file."""
        new_file = self.File(path=path, checksum=checksum, to_archive=to_archive, tags=tags)
        return new_file
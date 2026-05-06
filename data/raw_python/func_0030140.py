def reader(self):
        from ambry.orm.exc import NotFoundError
        from fs.errors import ResourceNotFoundError
        """The reader for the datafile"""

        try:
            return self.datafile.reader
        except ResourceNotFoundError:
            raise NotFoundError("Failed to find partition file, '{}' "
                                .format(self.datafile.path))
def local_datafile(self):
        """Return the datafile for this partition, from the build directory, the remote, or the warehouse"""
        from ambry_sources import MPRowsFile
        from fs.errors import ResourceNotFoundError
        from ambry.orm.exc import NotFoundError

        try:
            return MPRowsFile(self._bundle.build_fs, self.cache_key)

        except ResourceNotFoundError:
            raise NotFoundError(
                'Could not locate data file for partition {} (local)'.format(self.identity.fqname))
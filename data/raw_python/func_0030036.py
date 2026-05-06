def datafile(self):
        """Return an MPR datafile from the /ingest directory of the build filesystem"""
        from ambry_sources import MPRowsFile

        if self._datafile is None:
            if self.urltype == 'partition':
                    self._datafile = self.partition.datafile
            else:
                self._datafile = MPRowsFile(self._bundle.build_ingest_fs, self.name)

        return self._datafile
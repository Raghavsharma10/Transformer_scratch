def record_to_objects(self, preference=None):
        """Create objects from files, or merge the files into the objects. """
        from ambry.orm.file import File

        for f in self.list_records():

            pref = preference if preference else f.record.preference

            if pref == File.PREFERENCE.FILE:
                self._bundle.logger.debug('   Cleaning objects for file {}'.format(f.path))
                f.clean_objects()

            if pref in (File.PREFERENCE.FILE, File.PREFERENCE.MERGE):
                self._bundle.logger.debug('   rto {}'.format(f.path))
                f.record_to_objects()
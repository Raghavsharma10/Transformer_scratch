def objects_to_record(self, preference=None):
        """Create file records from objects. """
        from ambry.orm.file import File

        raise NotImplementedError("Still uses obsolete file_info_map")
        for file_const, (file_name, clz) in iteritems(file_info_map):
            f = self.file(file_const)

            pref = preference if preference else f.record.preference

            if pref in (File.PREFERENCE.MERGE, File.PREFERENCE.OBJECT):
                self._bundle.logger.debug('   otr {}'.format(file_const))
                f.objects_to_record()
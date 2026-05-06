def dict(self):
        """A dict that holds key/values for all of the properties in the
        object.

        :return:

        """
        SKIP_KEYS = ('_source_table', '_dest_table', 'd_vid', 't_vid', 'st_id',
                     'dataset', 'hash', 'process_records')
        return OrderedDict([(k, getattr(self, k)) for k in self.properties if k not in SKIP_KEYS])
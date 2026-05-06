def _write_log(self, version_key, meta_data, index_fields):
        """
        Creates a log entry for current object,
        Args:
            version_key(str): Version_bucket key from _write_version().
            meta_data (dict): JSON serializable meta data for logging of save operation.
                {'lorem': 'ipsum', 'dolar': 5}
            index_fields (list): Tuple list for secondary indexing keys in riak (with 'bin' or 'int').
                [('lorem','bin'),('dolar','int')]

        Returns:

        """
        meta_data = meta_data or {}
        meta_data.update({
            'version_key': version_key,
            'timestamp': time.time(),
        })
        obj = log_bucket.new(data=meta_data)
        obj.add_index('version_key_bin', version_key)
        obj.add_index('timestamp_int', int(meta_data['timestamp']))
        for field, index_type in index_fields:
            obj.add_index('%s_%s' % (field, index_type), meta_data.get(field, ""))
        obj.store()
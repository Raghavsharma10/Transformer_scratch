def _is_empty_cache_record(self, rec):
        """
        Return True if the specified cache record has no data, False otherwise.

        :param rec: cache record returned by :py:meth:`~._cache_get`
        :type rec: dict
        :return: True if record is empty, False otherwise
        :rtype: bool
        """
        # these are taken from DataQuery.query_one_table()
        for k in [
            'by_version',
            'by_file_type',
            'by_installer',
            'by_implementation',
            'by_system',
            'by_distro',
            'by_country'
        ]:
            if k in rec and len(rec[k]) > 0:
                return False
        return True
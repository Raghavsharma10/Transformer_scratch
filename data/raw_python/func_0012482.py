def _get_newest_ts_in_table(self, table_name):
        """
        Return the timestamp for the newest record in the given table.

        :param table_name: name of the table to query
        :type table_name: str
        :return: timestamp of newest row in table
        :rtype: int
        """
        logger.debug(
            'Querying for newest timestamp in table %s', table_name
        )
        q = "SELECT TIMESTAMP_TO_SEC(MAX(timestamp)) AS max_ts %s;" % (
            self._from_for_table(table_name)
        )
        res = self._run_query(q)
        ts = int(res[0]['max_ts'])
        logger.debug('Newest timestamp in table %s: %s', table_name, ts)
        return ts
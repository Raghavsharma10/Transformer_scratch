def query_one_table(self, table_name):
        """
        Run all queries for the given table name (date) and update the cache.

        :param table_name: table name to query against
        :type table_name: str
        """
        table_date = self._datetime_for_table_name(table_name)
        logger.info('Running all queries for date table: %s (%s)', table_name,
                    table_date.strftime('%Y-%m-%d'))
        final = self._dict_for_projects()
        try:
            data_timestamp = self._get_newest_ts_in_table(table_name)
        except HttpError as exc:
            try:
                content = json.loads(exc.content.decode('utf-8'))
                if content['error']['message'].startswith('Not found: Table'):
                    logger.error("Table %s not found; no data for that day",
                                 table_name)
                    return
            except:
                pass
            raise exc
        # data queries
        # note - ProjectStats._is_empty_cache_record() needs to know keys
        for name, func in {
            'by_version': self._query_by_version,
            'by_file_type': self._query_by_file_type,
            'by_installer': self._query_by_installer,
            'by_implementation': self._query_by_implementation,
            'by_system': self._query_by_system,
            'by_distro': self._query_by_distro,
            'by_country': self._query_by_country_code
        }.items():
            tmp = func(table_name)
            for proj_name in tmp:
                final[proj_name][name] = tmp[proj_name]
        # add to cache
        for proj_name in final:
            self.cache.set(proj_name, table_date, final[proj_name],
                           data_timestamp)
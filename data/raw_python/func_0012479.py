def _get_download_table_ids(self):
        """
        Get a list of PyPI downloads table (sharded per day) IDs.

        :return: list of table names (strings)
        :rtype: ``list``
        """
        all_table_names = []  # matching per-date table names
        logger.info('Querying for all tables in dataset')
        tables = self.service.tables()
        request = tables.list(projectId=self._PROJECT_ID,
                              datasetId=self._DATASET_ID)
        while request is not None:
            response = request.execute()
            # if the number of results is evenly divisible by the page size,
            # we may end up with a last response that has no 'tables' key,
            # and is empty.
            if 'tables' not in response:
                response['tables'] = []
            for table in response['tables']:
                if table['type'] != 'TABLE':
                    logger.debug('Skipping %s (type=%s)',
                                 table['tableReference']['tableId'],
                                 table['type'])
                    continue
                if not self._table_re.match(table['tableReference']['tableId']):
                    logger.debug('Skipping table with non-matching name: %s',
                                 table['tableReference']['tableId'])
                    continue
                all_table_names.append(table['tableReference']['tableId'])
            request = tables.list_next(previous_request=request,
                                       previous_response=response)
        return sorted(all_table_names)
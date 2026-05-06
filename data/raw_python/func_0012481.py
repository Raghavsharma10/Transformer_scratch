def _run_query(self, query):
        """
        Run one query against BigQuery and return the result.

        :param query: the query to run
        :type query: str
        :return: list of per-row response dicts (key => value)
        :rtype: ``list``
        """
        query_request = self.service.jobs()
        logger.debug('Running query: %s', query)
        start = datetime.now()
        resp = query_request.query(
            projectId=self.project_id, body={'query': query}
        ).execute()
        duration = datetime.now() - start
        logger.debug('Query response (in %s): %s', duration, resp)
        if not resp['jobComplete']:
            logger.error('Error: query reported job not complete!')
        if int(resp['totalRows']) == 0:
            return []
        if int(resp['totalRows']) != len(resp['rows']):
            logger.error('Error: query reported %s total rows, but only '
                         'returned %d', resp['totalRows'], len(resp['rows']))
        data = []
        fields = [f['name'] for f in resp['schema']['fields']]
        for row in resp['rows']:
            d = {}
            for idx, val in enumerate(row['f']):
                d[fields[idx]] = val['v']
            data.append(d)
        return data
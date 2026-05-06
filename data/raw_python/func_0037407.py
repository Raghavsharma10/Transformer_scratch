def _batch_get_archive_listing(self, archive_names):
        '''
        Batched version of :py:meth:`~DynamoDBManager._get_archive_listing`

        Returns a list of full archive listings from an iterable of archive
        names

        .. note ::

            Invalid archive names will simply not be returned, so the response
            may not be the same length as the supplied `archive_names`.

        Parameters
        ----------

        archive_names : list

            List of archive names

        Returns
        -------

        archive_listings : list

            List of archive listings

        '''

        archive_names = list(archive_names)

        MAX_QUERY_LENGTH = 100

        archives = []

        for query_index in range(0, len(archive_names), MAX_QUERY_LENGTH):
            current_query = {
                'Keys': [{'_id': i} for i in archive_names[
                    query_index: query_index+MAX_QUERY_LENGTH]]}

            attempts = 0
            res = {}

            while True:

                if attempts > 0 and len(res.get('UnprocessedKeys', {})) > 0:
                    current_query = res['UnprocessedKeys'][self._table_name]

                elif attempts > 0 and len(res.get('UnprocessedKeys', {})) == 0:
                    break

                res = self._resource.batch_get_item(
                    RequestItems={self._table_name: current_query})

                archives.extend(res['Responses'][self._table_name])

                attempts += 1

        return archives
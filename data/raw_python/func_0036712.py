def bulk(self, actions, stats_only=False, **kwargs):
        """
        Executes bulk api by elasticsearch.helpers.bulk.

        :param actions: iterator containing the actions
        :param stats_only:if `True` only report number of successful/failed
        operations instead of just number of successful and a list of error responses
        Any additional keyword arguments will be passed to
        :func:`~elasticsearch.helpers.streaming_bulk` which is used to execute
        the operation, see :func:`~elasticsearch.helpers.streaming_bulk` for more
        accepted parameters.
        """
        success, failed = es_helpers.bulk(self.client, actions, stats_only, **kwargs)
        logger.info('Bulk is done success %s failed %s actions: \n %s' % (success, failed, actions))
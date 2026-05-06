def create(self, stream, start, parameters, sources, end=None):
        """ Create a hitorics preview job.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/previewcreate

            :param stream: hash of the CSDL filter to create the job for
            :type stream: str
            :param start: Unix timestamp for the start of the period
            :type start: int
            :param parameters: list of historics preview parameters, can be found at http://dev.datasift.com/docs/api/rest-api/endpoints/previewcreate
            :type parameters: list
            :param sources: list of sources to include, eg. ['tumblr','facebook']
            :type sources: list
            :param end: (optional) Unix timestamp for the end of the period, defaults to min(start+24h, now-1h)
            :type end: int
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.HistoricSourcesRequired`, :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        if len(sources) == 0:
            raise HistoricSourcesRequired()
        if isinstance(sources, six.string_types):
            sources = [sources]
        params = {'hash': stream, 'start': start, 'sources': ','.join(sources), 'parameters': ','.join(parameters)}
        if end:
            params['end'] = end
        return self.request.post('create', params)
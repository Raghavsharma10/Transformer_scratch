def prepare(self, hash, start, end, name, sources, sample=None):
        """ Prepare a historics query which can later be started.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/historicsprepare

            :param hash: The hash of a CSDL create the query for
            :type hash: str
            :param start: when to start querying data from - unix timestamp
            :type start: int
            :param end: when the query should end - unix timestamp
            :type end: int
            :param name: the name of the query
            :type name: str
            :param sources: list of sources  e.g. ['facebook','bitly','tumblr']
            :type sources: list
            :param sample: percentage to sample, either 10 or 100
            :type sample: int
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.HistoricSourcesRequired`, :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        if len(sources) == 0:
            raise HistoricSourcesRequired()
        if not isinstance(sources, list):
            sources = [sources]

        params = {'hash': hash, 'start': start, 'end': end, 'name': name, 'sources': ','.join(sources)}
        if sample:
            params['sample'] = sample
        return self.request.post('prepare', params)
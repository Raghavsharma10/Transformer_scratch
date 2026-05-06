def pull(self, subscription_id, size=None, cursor=None):
        """ Pulls a series of interactions from the queue for the given subscription ID.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/pull

            :param subscription_id: The ID of the subscription to pull interactions for
            :type subscription_id: str
            :param size: the max amount of data to pull in bytes
            :type size: int
            :param cursor: an ID to use as the point in the queue from which to start fetching data
            :type cursor: str
            :returns: dict with extra response data
            :rtype: :class:`~datasift.request.ResponseList`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        params = {'id': subscription_id}
        if size:
            params['size'] = size
        if cursor:
            params['cursor'] = cursor
        raw = self.request('get', 'pull', params=params)

        def pull_parser(headers, data):
            pull_type = headers.get("X-DataSift-Format")
            if pull_type in ("json_meta", "json_array"):
                return json.loads(data)
            else:
                lines = data.strip().split("\n").__iter__()
                return list(map(json.loads, lines))

        return self.request.build_response(raw, parser=pull_parser)
def get(self, suffix, params=None):
        """
        request weibo api
        :param suffix: str,
        :param params: dict, url query parameters
        :return:

        """

        url = self.base + suffix
        params = filter_params(params)

        response = self.session.get(url=url, params=params)

        return self._handler_response(response)
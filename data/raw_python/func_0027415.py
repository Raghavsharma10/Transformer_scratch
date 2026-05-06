def post(self, suffix, params=None, data=None, files=None):
        """
        :return:
        """

        url = self.base + suffix
        params = filter_params(params)

        response = self.session.post(url=url, params=params, data=data, files=files)

        return self._handler_response(response, data=data)
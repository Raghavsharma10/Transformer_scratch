def post(self, resource_path: str, data: dict, administration_id: int = None):
        """
        Performs a POST request to the endpoint identified by the resource path. POST requests are usually used to add
        new data.

        Example:
            >>> from moneybird import MoneyBird, TokenAuthentication
            >>> moneybird = MoneyBird(TokenAuthentication('access_token'))
            >>> data = {'url': 'http://www.mocky.io/v2/5185415ba171ea3a00704eed'}
            >>> moneybird.post('webhooks', data, 123)
            {'id': '143274315994891267', 'url': 'http://www.mocky.io/v2/5185415ba171ea3a00704eed', ...

        :param resource_path: The resource path.
        :param data: The data to send to the server.
        :param administration_id: The administration id (optional, depending on the resource path).
        :return: The decoded JSON response for the request.
        """
        response = self.session.post(
            url=self._get_url(administration_id, resource_path),
            json=data,
        )
        return self._process_response(response)
async def request(self, method, path, json=None):
        """Make a request to the API."""
        url = 'https://{}:{}/api/'.format(self.host, self.port)
        url += path.format(site=self.site)

        try:
            async with self.session.request(method, url, json=json) as res:
                if res.content_type != 'application/json':
                    raise ResponseError(
                        'Invalid content type: {}'.format(res.content_type))
                response = await res.json()
                _raise_on_error(response)
                return response['data']

        except client_exceptions.ClientError as err:
            raise RequestError(
                'Error requesting data from {}: {}'.format(self.host, err)
            ) from None
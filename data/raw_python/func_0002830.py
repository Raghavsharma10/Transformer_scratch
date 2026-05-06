async def _request(
            self, method: str, endpoint: str, *, headers: dict = None) -> dict:
        """Make a request against air-matters.com."""
        url = '{0}/{1}'.format(API_URL_SCAFFOLD, endpoint)

        if not headers:
            headers = {}
        headers.update({
            'Host': DEFAULT_HOST,
            'Origin': DEFAULT_ORIGIN,
            'Referer': DEFAULT_ORIGIN,
            'User-Agent': DEFAULT_USER_AGENT,
        })

        async with self._websession.request(
                method,
                url,
                headers=headers,
        ) as resp:
            try:
                resp.raise_for_status()
                return await resp.json(content_type=None)
            except client_exceptions.ClientError as err:
                raise RequestError(
                    'Error requesting data from {0}: {1}'.format(
                        endpoint, err)) from None
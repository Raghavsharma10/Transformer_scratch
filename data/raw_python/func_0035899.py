def _call(self, method, *args, **kwargs):
        """Call the remote service and return the response data."""

        assert self.session

        if not kwargs.get('verify'):
            kwargs['verify'] = self.SSL_VERIFY

        response = self.session.request(method, *args, **kwargs)
        response_json = response.text and response.json() or {}

        if response.status_code < 200 or response.status_code >= 300:
            message = response_json.get('error', response_json.get('message'))
            raise HelpScoutRemoteException(response.status_code, message)

        self.page_current = response_json.get(self.PAGE_CURRENT, 1)
        self.page_total = response_json.get(self.PAGE_TOTAL, 1)

        try:
            return response_json[self.PAGE_DATA_MULTI]
        except KeyError:
            pass

        try:
            return [response_json[self.PAGE_DATA_SINGLE]]
        except KeyError:
            pass

        return None
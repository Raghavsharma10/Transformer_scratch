def _get(self, resource, payload=None):
        ''' Wrapper around requests.get that shorten caller url and takes care
        of errors '''
        # Avoid dangerous default function argument `{}`
        payload = payload or {}
        # Build the request and return json response
        return requests.get(
            '{}/{}/{}'.format(
                self.master, pyconsul.__consul_api_version__, resource),
            params=payload
        )
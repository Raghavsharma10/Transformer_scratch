def _do_sse_request(self, path, params=None):
        """Query Marathon server for events."""
        urls = [''.join([server.rstrip('/'), path]) for server in self.servers]
        while urls:
            url = urls.pop()
            try:
                # Requests does not set the original Authorization header on cross origin
                # redirects. If set allow_redirects=True we may get a 401 response.
                response = self.sse_session.get(
                    url,
                    params=params,
                    stream=True,
                    headers={'Accept': 'text/event-stream'},
                    auth=self.auth,
                    verify=self.verify,
                    allow_redirects=False
                )
            except Exception as e:
                marathon.log.error(
                    'Error while calling %s: %s', url, e.message)
            else:
                if response.is_redirect and response.next:
                    urls.append(response.next.url)
                    marathon.log.debug("Got redirect to {}".format(response.next.url))
                elif response.ok:
                    return response.iter_lines()

        raise MarathonError('No remaining Marathon servers to try')
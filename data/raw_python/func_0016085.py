def _do_request(self, method, path, params=None, data=None):
        """Query Marathon server."""
        headers = {
            'Content-Type': 'application/json', 'Accept': 'application/json'}

        if self.auth_token:
            headers['Authorization'] = "token={}".format(self.auth_token)

        response = None
        servers = list(self.servers)
        while servers and response is None:
            server = servers.pop(0)
            url = ''.join([server.rstrip('/'), path])
            try:
                response = self.session.request(
                    method, url, params=params, data=data, headers=headers,
                    auth=self.auth, timeout=self.timeout, verify=self.verify)
                marathon.log.info('Got response from %s', server)
            except requests.exceptions.RequestException as e:
                marathon.log.error(
                    'Error while calling %s: %s', url, str(e))

        if response is None:
            raise NoResponseError('No remaining Marathon servers to try')

        if response.status_code >= 500:
            marathon.log.error('Got HTTP {code}: {body}'.format(
                code=response.status_code, body=response.text.encode('utf-8')))
            raise InternalServerError(response)
        elif response.status_code >= 400:
            marathon.log.error('Got HTTP {code}: {body}'.format(
                code=response.status_code, body=response.text.encode('utf-8')))
            if response.status_code == 404:
                raise NotFoundError(response)
            elif response.status_code == 409:
                raise ConflictError(response)
            else:
                raise MarathonHttpError(response)
        elif response.status_code >= 300:
            marathon.log.warn('Got HTTP {code}: {body}'.format(
                code=response.status_code, body=response.text.encode('utf-8')))
        else:
            marathon.log.debug('Got HTTP {code}: {body}'.format(
                code=response.status_code, body=response.text.encode('utf-8')))

        return response
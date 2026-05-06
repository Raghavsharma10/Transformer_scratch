def make_request(self, path, params=None, data=None, method=None):
        '''
        Makes a request to the cheddar api using the authentication and
        configuration settings available.
        '''
        # Setup values
        url = self.build_url(path, params)
        client_log.debug('Requesting:  %s' % url)
        method = method or 'GET'
        body = None
        headers = {}

        if data:
            method = 'POST'
            body = urlencode(data)
            headers = {
                'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
            }

        client_log.debug('Request Method:  %s' % method)
        client_log.debug('Request Body(Data):  %s' % data)
        client_log.debug('Request Body(Raw):  %s' % body)

        # Setup http client
        h = httplib2.Http(cache=self.cache, timeout=self.timeout)
        #h.add_credentials(self.username, self.password)
        # Skip the normal http client behavior and send auth headers immediately
        # to save an http request.
        headers['Authorization'] = "Basic %s" % base64.standard_b64encode(self.username + ':' + self.password).strip()

        # Make request
        response, content = h.request(url, method, body=body, headers=headers)
        status = response.status
        client_log.debug('Response Status:  %d' % status)
        client_log.debug('Response Content:  %s' % content)
        if status != 200 and status != 302:
            exception_class = CheddarError
            if status == 401:
                exception_class = AccessDenied
            elif status == 400:
                exception_class = BadRequest
            elif status == 404:
                exception_class = NotFound
            elif status == 412:
                exception_class = PreconditionFailed
            elif status == 500:
                exception_class = CheddarFailure
            elif status == 502:
                exception_class = NaughtyGateway
            elif status == 422:
                exception_class = UnprocessableEntity

            raise exception_class(response, content)

        response.content = content
        return response
def request(self, uri, method=GET, headers=None, cookies=None, params=None, data=None, post_files=None,**kwargs):
        """Makes a request using requests

        @param uri: The uri to send request
        @param method: Method to use to send request
        @param headers: Any headers to send with request
        @param cookies: Request cookies (in addition to session cookies)
        @param params: Request parameters
        @param data: Request data
        @param kwargs: other options to pass to underlying request
        @rtype: requests.Response
        @return: The response
        """

        coyote_args = {
            'headers': headers,
            'cookies': cookies,
            'params': params,
            'files': post_files,
            'data': data,
            'verify': self.verify_certificates,

        }

        coyote_args.update(kwargs)

        if method == self.POST:
            response = self.session.post(uri, **coyote_args)

        elif method == self.PUT:
            response = self.session.put(uri, **coyote_args)

        elif method == self.PATCH:
            response = self.session.patch(uri, **coyote_args)

        elif method == self.DELETE:
            response = self.session.delete(uri, **coyote_args)

        else:  # Default to GET
            response = self.session.get(uri, **coyote_args)

        self.responses.append(response)

        while len(self.responses) > self.max_response_history:
            self.responses.popleft()

        return response
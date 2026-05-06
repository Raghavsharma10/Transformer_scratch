def request(self, url, parameters):
        """Perform a http(s) request for given parameters to given URL

            Keyword arguments:
            url -- API url
            parameters -- dict with payload.
        """

        try:
            request = Request(url + '?' + urlencode(parameters), None, {
                'X-Authentication-Token': self.api_key,
                'User-Agent': self.user_agent
                },
                None,
                False,
                "GET"
            )

            response = urlopen(request)
        except HTTPError as e:
            self.logger.error('Code: ' + str(e.code))
            self.logger.error('Response: ' + str(e.reason))

            if e.code == 400:
                raise UnknownOutputFormatException()

            if e.code == 401:
                raise AuthorizationRequiredException()

            if e.code == 402:
                raise NotEnoughCreditsException()

            if e.code == 403:
                raise AccessDeniedException()

            if e.code == 422:
                raise InvalidMacOrOuiException()

            if e.code >= 500:
                raise ServerErrorException("Response code: {}".format(e.code))

            raise ServerErrorException(e.reason)

        if response.code >= 300 or response.code < 200:
            raise ServerErrorException("Response code: {}".format(response.code))

        headers = dict(response.getheaders())
        if "Warning" in headers.keys():
            self.logger.warning(headers["Warning"])

        self.logger.debug(response.info())

        return response.read()
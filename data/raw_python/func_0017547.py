def _request(self, capability, options=None, stream=False):
        """
        Make a _request to the RETS server
        :param capability: The name of the capability to use to get the URI
        :param options: Options to put into the _request
        :return: Response
        """
        if options is None:
            options = {}

        options.update({
            'headers': self.client.headers.copy()
        })

        url = self.capabilities.get(capability)

        if not url:
            msg = "{0!s} tried but no valid endpoints was found. Did you forget to Login?".format(capability)
            raise NotLoggedIn(msg)

        if self.user_agent_password:
            ua_digest = self._user_agent_digest_hash()
            options['headers']['RETS-UA-Authorization'] = 'Digest {0!s}'.format(ua_digest)

        if self.use_post_method and capability != 'Action':  # Action Requests should always be GET
            query = options.get('query')
            response = self.client.post(url, data=query, headers=options['headers'], stream=stream)
        else:
            if 'query' in options:
                url += '?' + '&'.join('{0!s}={1!s}'.format(k, quote(str(v))) for k, v in options['query'].items())

            response = self.client.get(url, headers=options['headers'], stream=stream)

        if response.status_code in [400, 401]:
            if capability == 'Login':
                m = "Could not log into the RETS server with the provided credentials."
            else:
                m = "The RETS server returned a 401 status code. You must be logged in to make this request."
            raise NotLoggedIn(m)

        elif response.status_code == 404 and self.use_post_method:
            raise HTTPException("Got a 404 when making a POST request. Try setting use_post_method=False when "
                                "initializing the Session.")

        return response
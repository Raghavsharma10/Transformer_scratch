def call(self, verb, servicePath, data=None, headers=None, forceText=False, sendJson=True):
        """Call the Nutch Server, do some error checking, and return the response.

        :param verb: One of nutch.RequestVerbs
        :param servicePath: path component of URL to append to endpoint, e.g. '/config'
        :param data: Data to attach to this request
        :param headers: headers to attach to this request, default are JsonAcceptHeader
        :param forceText: don't trust the response headers and just get the text
        :param sendJson: Whether to treat attached data as JSON or not
        """

        default_data = {} if sendJson else ""
        data = data if data else default_data

        headers = headers if headers else JsonAcceptHeader.copy()

        if not sendJson:
            headers.update(TextSendHeader)

        if verb not in RequestVerbs:
            die('Server call verb must be one of %s' % str(RequestVerbs.keys()))
        if Verbose:
            echo2("%s Endpoint:" % verb.upper(), servicePath)
            echo2("%s Request data:" % verb.upper(), data)
            echo2("%s Request headers:" % verb.upper(), headers)
        verbFn = RequestVerbs[verb]

        if sendJson:
            resp = verbFn(self.serverEndpoint + servicePath, json=data, headers=headers)
        else:
            resp = verbFn(self.serverEndpoint + servicePath, data=data, headers=headers)

        if Verbose:
            echo2("Response headers:", resp.headers)
            echo2("Response status:", resp.status_code)
        if resp.status_code != 200:
            if self.raiseErrors:
                error = NutchException("Unexpected server response: %d" % resp.status_code)
                error.status_code = resp.status_code
                raise error
            else:
                warn('Nutch server returned status:', resp.status_code)
        if forceText or 'content-type' not in resp.headers or resp.headers['content-type'] == 'text/plain':
            if Verbose:
                echo2("Response text:", resp.text)
            return resp.text

        content_type = resp.headers['content-type']
        if content_type == 'application/json' and not forceText:
            if Verbose:
                echo2("Response JSON:", resp.json())
            return resp.json()
        else:
            die('Did not understand server response: %s' % resp.headers)
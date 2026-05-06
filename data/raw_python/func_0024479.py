def process_request(self, req, resp):
        """
        Do response processing
        """
        # req.stream corresponds to the WSGI wsgi.input environ variable,
        # and allows you to read bytes from the request body.
        #
        # See also: PEP 3333
        if req.content_length in (None, 0):
            # Nothing to do
            req.context['data'] = req.params.copy()
            req.context['result'] = {}
            return
        else:
            req.context['result'] = {}

        body = req.stream.read()
        if not body:
            raise falcon.HTTPBadRequest('Empty request body',
                                        'A valid JSON document is required.')

        try:
            json_data = body.decode('utf-8')
            req.context['data'] = json.loads(json_data)
            try:
                log.info("REQUEST DATA: %s" % json_data)
            except:
                log.exception("ERR: REQUEST DATA CANT BE LOGGED ")
        except (ValueError, UnicodeDecodeError):
            raise falcon.HTTPError(falcon.HTTP_753,
                                   'Malformed JSON',
                                   'Could not decode the request body. The '
                                   'JSON was incorrect or not encoded as '
                                   'UTF-8.')
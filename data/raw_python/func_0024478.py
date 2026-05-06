def process_request(self, req, resp):
        """
        Do response processing
        """
        if not req.client_accepts_json:
            raise falcon.HTTPNotAcceptable(
                'This API only supports responses encoded as JSON.',
                href='http://docs.examples.com/api/json')
        if req.method in ('POST', 'PUT'):
            if req.content_length != 0 and \
                            'application/json' not in req.content_type and \
                            'text/plain' not in req.content_type:
                raise falcon.HTTPUnsupportedMediaType(
                    'This API only supports requests encoded as JSON.',
                    href='http://docs.examples.com/api/json')
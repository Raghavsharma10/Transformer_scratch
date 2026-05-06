def mk_req(self, url, **kwargs):
        """
        Helper function to create a tornado HTTPRequest object, kwargs get passed in to
        create the HTTPRequest object. See:
        http://tornado.readthedocs.org/en/latest/httpclient.html#request-objects
        """
        req_url = self.base_url + url
        req_kwargs = kwargs
        req_kwargs['ca_certs'] = req_kwargs.get('ca_certs', self.certs)
        # have to do this because tornado's HTTP client doesn't
        # play nice with elasticsearch
        req_kwargs['allow_nonstandard_methods'] = req_kwargs.get(
            'allow_nonstandard_methods',
            True
        )
        return HTTPRequest(req_url, **req_kwargs)
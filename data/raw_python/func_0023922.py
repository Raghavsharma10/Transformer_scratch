def dispatch(self, request, **kwargs):
        '''
        Entry point for this class, here we decide basic stuff
        '''

        # Check if this is a webservice request
        self.json_worker = (bool(getattr(self.request, "authtoken", False))) or (self.json is True)
        self.__authtoken = (bool(getattr(self.request, "authtoken", False)))

        # Check if this is an AJAX request
        if (request.is_ajax() or self.json_worker) and request.body:
            request.POST = QueryDict('').copy()
            body = request.body
            if type(request.body) == bytes:
                body = body.decode("utf-8")
            post = json.loads(body)
            for key in post:
                if type(post[key]) == dict and '__JSON_DATA__' in post[key]:
                    post[key] = json.dumps(post[key]['__JSON_DATA__'])

            request.POST.update(post)

        # Set class internal variables
        self._setup(request)

        # Call the base implementation
        return super(GenModify, self).dispatch(request, **kwargs)
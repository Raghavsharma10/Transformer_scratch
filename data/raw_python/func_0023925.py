def dispatch(self, request, **kwargs):
        '''
        Entry point for this class, here we decide basic stuff
        '''

        # Check if this is a REST query to pusth the answer to responde in JSON
        if bool(self.request.META.get('HTTP_X_REST', False)):
            self.json = True

        # Check if this is a REST query to add an element
        if self.request.method in ['PUT', 'DELETE']:
            if self.request.method == 'PUT':
                action = 'edit'
            else:
                action = 'delete'

            # Set new method
            self.request.method == 'POST'

            # Find the URL
            target = get_class(resolve("{}/{}".format(self.request.META.get("REQUEST_URI"), action)).func)

            # Make sure we will answer as an API
            target.json = True

            # Lets go for it
            return target.as_view()(self.request, pk=kwargs.get('pk'))

        # Detect if we have to answer in json
        self.__authtoken = (bool(getattr(self.request, "authtoken", False)))
        self.json_worker = self.__authtoken or (self.json is True)

        # Check if this is an AJAX request
        if (request.is_ajax() or self.json_worker) and request.body:
            request.POST = json.loads(request.body)

        # Set class internal variables
        self._setup(request)

        # Call the base implementation
        return super(GenDetail, self).dispatch(request, **kwargs)
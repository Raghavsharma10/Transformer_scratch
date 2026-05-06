def _setup(self, request):
        '''
        Entry point for this class, here we decide basic stuff
        '''

        # Get details from self
        info = model_inspect(self)
        self._appname = getattr(self, 'appname', info['appname'])
        self._modelname = getattr(self, 'modelname', info['modelname'])

        # Get user information
        if not hasattr(self, 'user'):
            self.user = self.request.user
        # Get profile
        self.profile = get_profile(self.user)

        # Get language
        self.language = get_language()

        # Default value for no foreign key attribute
        if 'no_render_as_foreign' not in self.extra_context:
            self.extra_context['no_render_as_foreign'] = []
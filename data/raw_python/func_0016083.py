def load_configuration(self, **kwargs):
        '''load configuration, merge with default settings'''
        # update passed arguments with default values
        for key in settings.ACTIVE_URL_KWARGS:
            kwargs.setdefault(key, settings.ACTIVE_URL_KWARGS[key])

        # "active" html tag css class
        self.css_class = kwargs['css_class']
        # "active" html tag
        self.parent_tag = kwargs['parent_tag']
        # flipper for menu support
        self.menu = kwargs['menu']
        # whether to ignore / chomp get_params
        self.ignore_params = kwargs['ignore_params']
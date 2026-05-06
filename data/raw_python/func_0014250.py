def register_app(self, app=None):
        '''
        Registers an app as a "DMP-enabled" app.  Normally, DMP does this
        automatically when included in urls.py.

        If app is None, the DEFAULT_APP is registered.
        '''
        app = app or self.options['DEFAULT_APP']
        if not app:
            raise ImproperlyConfigured('An app name is required because DEFAULT_APP is empty - please use a '
                                        'valid app name or set the DEFAULT_APP in settings')
        if isinstance(app, str):
            app = apps.get_app_config(app)

        # since this only runs at startup, this lock doesn't affect performance
        with self.registration_lock:
            # short circuit if already registered
            if app.name in self.registered_apps:
                return

            # first time for this app, so add to our dictionary
            self.registered_apps[app.name] = app

            # set up the template, script, and style renderers
            # these create and cache just by accessing them
            self.engine.get_template_loader(app, 'templates', create=True)
            self.engine.get_template_loader(app, 'scripts', create=True)
            self.engine.get_template_loader(app, 'styles', create=True)

            # send the registration signal
            if self.options['SIGNALS']:
                dmp_signal_register_app.send(sender=self, app_config=app)
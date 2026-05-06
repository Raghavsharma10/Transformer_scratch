def register(self, app, *args, **kwargs):
        " Activate loginmanager and principal. "

        if not self._login_manager or self.app != app:
            self._login_manager = LoginManager()
            self._login_manager.user_callback = self.user_loader
            self._login_manager.setup_app(app)
            self._login_manager.login_view = 'urls.index'
            self._login_manager.login_message = u'You need to be signed in for this page.'

        self.app = app

        if not self._principal:
            self._principal = Principal(app)
            identity_loaded.connect(self.identity_loaded)

        super(UserManager, self).register(app, *args, **kwargs)
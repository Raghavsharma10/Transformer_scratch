def init_app(self, app, session_kvstore=None):
        """Initialize application and KVSession.

        This will replace the session management of the application with
        Flask-KVSession's.

        :param app: The :class:`~flask.Flask` app to be initialized."""
        app.config.setdefault('SESSION_KEY_BITS', 64)
        app.config.setdefault('SESSION_RANDOM_SOURCE', SystemRandom())

        if not session_kvstore and not self.default_kvstore:
            raise ValueError('Must supply session_kvstore either on '
                             'construction or init_app().')

        # set store on app, either use default
        # or supplied argument
        app.kvsession_store = session_kvstore or self.default_kvstore

        app.session_interface = KVSessionInterface()
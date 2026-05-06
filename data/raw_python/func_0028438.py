def init_app(self, app):
        """
        Initializes the extension with the given app, registers the
        built-in views with an own blueprint and hooks up our signal
        callbacks.
        """
        # If no version path was provided in the init of the Dockerflow
        # class we'll use the parent directory of the app root path.
        if self.version_path is None:
            self.version_path = os.path.dirname(app.root_path)

        for view in (
            ('/__version__', 'version', self._version_view),
            ('/__heartbeat__', 'heartbeat', self._heartbeat_view),
            ('/__lbheartbeat__', 'lbheartbeat', self._lbheartbeat_view),
        ):
            self._blueprint.add_url_rule(*view)
        self._blueprint.before_app_request(self._before_request)
        self._blueprint.after_app_request(self._after_request)
        self._blueprint.app_errorhandler(HeartbeatFailure)(self._heartbeat_exception_handler)
        app.register_blueprint(self._blueprint)
        got_request_exception.connect(self._got_request_exception, sender=app)

        if not hasattr(app, 'extensions'):  # pragma: nocover
            app.extensions = {}
        app.extensions['dockerflow'] = self
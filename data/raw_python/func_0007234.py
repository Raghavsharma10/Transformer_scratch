def spawn_api(self, app, decorator=None):
        """Auto-generate server endpoints implementing the API into this Flask app"""
        if decorator:
            assert type(decorator).__name__ == 'function'
        self.is_server = True
        self.app = app

        if self.local:
            # Re-generate client callers, this time as local and passing them the app
            self._generate_client_callers(app)

        return spawn_server_api(self.name, app, self.api_spec, self.error_callback, decorator)
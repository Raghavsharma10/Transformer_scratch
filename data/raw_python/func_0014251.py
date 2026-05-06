def is_registered_app(self, app):
        '''Returns true if the given app/app name is registered with DMP'''
        if app is None:
            return False
        if isinstance(app, AppConfig):
            app = app.name
        return app in self.registered_apps
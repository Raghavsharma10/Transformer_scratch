def get_glitter_app(self, glitter_app_name):
        """
        Retrieve the Glitter App config for a specific Glitter App.
        """
        if not self.discovered:
            self.discover_glitter_apps()

        try:
            glitter_app = self.glitter_apps[glitter_app_name]
            return glitter_app
        except KeyError:
            return None
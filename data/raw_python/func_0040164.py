def str_to_application_class(self, an_app_key):
        """a configman compatible str_to_* converter"""
        try:
            app_class = str_to_python_object(self.apps[an_app_key])
        except KeyError:
            app_class = str_to_python_object(an_app_key)
        try:
            self.application_defaults = DotDict(
                app_class.get_application_defaults()
            )
        except AttributeError:
            # no get_application_defaults, skip this step
            pass
        return app_class
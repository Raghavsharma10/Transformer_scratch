def get_apps(self):
        """
        Get the list of installed apps
        and return the apps that have
        an emails module
        """
        templates = []
        for app in settings.INSTALLED_APPS:
            try:
                app = import_module(app + '.emails')
                templates += self.get_plugs_mail_classes(app)
            except ImportError:
                pass
        return templates
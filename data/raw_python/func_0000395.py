def discover_glitter_apps(self):
        """
        Find all the Glitter App configurations in the current project.
        """
        for app_name in settings.INSTALLED_APPS:
            module_name = '{app_name}.glitter_apps'.format(app_name=app_name)
            try:
                glitter_apps_module = import_module(module_name)
                if hasattr(glitter_apps_module, 'apps'):
                    self.glitter_apps.update(glitter_apps_module.apps)
            except ImportError:
                pass

        self.discovered = True
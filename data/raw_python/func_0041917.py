def autodiscover(self, module_name=None, verbose=False):
        """Autodiscovers classes in the notifications.py file of any
        INSTALLED_APP.
        """
        module_name = module_name or "notifications"
        verbose = True if verbose is None else verbose
        sys.stdout.write(f" * checking for {module_name} ...\n")
        for app in django_apps.app_configs:
            try:
                mod = import_module(app)
                try:
                    before_import_registry = copy.copy(site_notifications._registry)
                    import_module(f"{app}.{module_name}")
                    if verbose:
                        sys.stdout.write(
                            f" * registered notifications from application '{app}'\n"
                        )
                except Exception as e:
                    if f"No module named '{app}.{module_name}'" not in str(e):
                        site_notifications._registry = before_import_registry
                        if module_has_submodule(mod, module_name):
                            raise
            except ModuleNotFoundError:
                pass
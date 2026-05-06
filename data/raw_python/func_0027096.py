def _get_app_version(self, app_config):
        """
        Some plugins ship multiple applications and extensions.
        However all of them have the same version, because they are released together.
        That's why only-top level module is used to fetch version information.
        """

        base_name = app_config.__module__.split('.')[0]
        module = __import__(base_name)
        return getattr(module, '__version__', 'N/A')
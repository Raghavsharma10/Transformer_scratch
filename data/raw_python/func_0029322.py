def get_driver(self, namespace, parsed_args, **kwargs):
        """Get mutually-exlusive plugin for plugin namespace.

        """
        option, dest = self._namespace_to_option(namespace)
        dest_prefix = '{0}_'.format(dest)
        driver_name = getattr(parsed_args, dest, 'default')
        driver_extension = self.driver_managers[namespace][driver_name]
        return driver_extension.plugin.from_args(
            parsed_args, dest_prefix, **kwargs)
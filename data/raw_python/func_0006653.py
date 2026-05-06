def resource_filename(self, resource_name, module_name=None):
        """Return a resource based on a file name.  This uses the ``pkg_resources``
        package first to find the resources.  If it doesn't find it, it returns
        a path on the file system.

        :param: resource_name the file name of the resource to obtain (or name
            if obtained from an installed module)
        :param module_name: the name of the module to obtain the data, which
            defaults to ``__name__``
        :return: a path on the file system or resource of the installed module

        """
        if module_name is None:
            mod = self._get_calling_module()
            logger.debug(f'calling module: {mod}')
            if mod is not None:
                mod_name = mod.__name__
        if module_name is None:
            module_name = __name__
        if pkg_resources.resource_exists(mod_name, resource_name):
            res = pkg_resources.resource_filename(mod_name, resource_name)
        else:
            res = resource_name
        return Path(res)
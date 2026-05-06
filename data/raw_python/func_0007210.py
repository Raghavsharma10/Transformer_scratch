def get_module_class(self):
        """Return the module and class as a tuple of the given class in the
        initializer.

        :param reload: if ``True`` then reload the module before returning the
        class

        """
        pkg, cname = self.parse_module_class()
        logger.debug(f'pkg: {pkg}, class: {cname}')
        pkg = pkg.split('.')
        mod = reduce(lambda m, n: getattr(m, n), pkg[1:], __import__(pkg[0]))
        logger.debug(f'mod: {mod}')
        if self.reload:
            importlib.reload(mod)
        cls = getattr(mod, cname)
        logger.debug(f'class: {cls}')
        return mod, cls
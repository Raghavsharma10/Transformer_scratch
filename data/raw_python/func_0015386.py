def load_module(self, name):
        """
        Only lets modules in allowed_modules be loaded, others
        will get an ImportError
        """

        # Get the name relative to SITEDIR ..
        filepath = self.module_info[1]
        fullname = splitext( \
            relpath(filepath, self.sitedir) \
            )[0].replace(os.sep, '.')

        modulename = filename_to_module(fullname)
        if modulename not in allowed_modules:
            if remember_blocks:
                blocked_imports.add(fullname)
            if log_blocks:
                raise ImportError("Vext blocked import of '%s'" % modulename)
            else:
                # Standard error message
                raise ImportError("No module named %s" % modulename)

        if name not in sys.modules:
            try:
                logger.debug("load_module %s %s", name, self.module_info)
                module = imp.load_module(name, *self.module_info)
            except Exception as e:
                logger.debug(e)
                raise
            sys.modules[fullname] = module

        return sys.modules[fullname]
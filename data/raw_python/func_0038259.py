def load_modules(self):
        """
        Locate and import modules from locations specified during initialization.

        Locations include:
            - Program's standard library (``library``)
            - `Entry points <Entry point_>`_ (``entry_point``)
            - Specified modules (``modules``)
            - Specified paths (``paths``)

        If a malformed child plugin class is imported, a :py:exc:`PluginWarning` will be issued,
        the class is skipped, and loading operations continue.

        If an invalid `entry point <Entry point_>`_ is specified, an :py:exc:`EntryPointWarning`
        is issued and loading operations continue.
        """

        # Start with standard library
        if self.library:
            LOGGER.info('Loading plugins from standard library')
            libmod = _import_module(self.library)
            _recursive_import(libmod)

        # Get entry points
        if self.entry_point:
            LOGGER.info('Loading plugins from entry points group %s', self.entry_point)
            for epoint in iter_entry_points(group=self.entry_point):
                try:
                    mod = _import_module(epoint)
                except PluginImportError as e:
                    warnings.warn("Module %s can not be loaded for entry point %s: %s" %
                                  (epoint.module_name, epoint.name, e), EntryPointWarning)
                    continue

                # If we have a package, walk it
                if ismodule(mod):
                    _recursive_import(mod)
                else:
                    warnings.warn("Entry point '%s' is not a module or package" % epoint.name,
                                  EntryPointWarning)

        # Load auxiliary modules
        if self.modules:
            for mod in self.modules:
                LOGGER.info('Loading plugins from %s', mod)
                _recursive_import(_import_module(mod))

        # Load auxiliary paths
        if self.paths:
            auth_paths_mod = importlib.import_module(self.prefix_package)
            initial_path = auth_paths_mod.__path__[:]

            # Append each path to module path
            for path in self.paths:

                modpath = os.path.realpath(path)
                if os.path.isdir(modpath):
                    LOGGER.info('Adding %s as a plugin search path', path)
                    if modpath not in auth_paths_mod.__path__:
                        auth_paths_mod.__path__.append(modpath)

                else:
                    LOGGER.info("Configured plugin path '%s' is not a valid directory", path)

            # Walk packages
            try:
                _recursive_import(auth_paths_mod)

            finally:
                # Restore Path
                auth_paths_mod.__path__[:] = initial_path

        self.loaded = True
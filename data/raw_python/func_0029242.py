def initialize_plugins(self):
        """Attempt to Load and initialize all the plugins.

        Any issues loading plugins will be output to stderr.

        """
        for plugin_name in self.options.plugin:
            parts = plugin_name.split('.')
            if len(parts) > 1:
                module_name = '.'.join(parts[:-1])
                class_name = parts[-1]
            else:
                # Use the titlecase format of the module name as the class name
                module_name = parts[0]
                class_name = parts[0].title()

            # First try to load plugins from the passed in plugins_dir and then
            # from the hairball.plugins package.
            plugin = None
            for package in (None, 'hairball.plugins'):
                if package:
                    module_name = '{}.{}'.format(package, module_name)
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    # Initializes the plugin by calling its constructor
                    plugin = getattr(module, class_name)()

                    # Verify plugin is of the correct class
                    if not isinstance(plugin, HairballPlugin):
                        sys.stderr.write('Invalid type for plugin {}: {}\n'
                                         .format(plugin_name, type(plugin)))
                        plugin = None
                    else:
                        break
                except (ImportError, AttributeError):
                    pass
            if plugin:
                self.plugins.append(plugin)
            else:
                sys.stderr.write('Cannot find plugin {}\n'.format(plugin_name))
        if not self.plugins:
            sys.stderr.write('No plugins loaded. Goodbye!\n')
            sys.exit(1)
def refresh_files(self):
        """
        Discovers the available plugins and turns each into a module object.
        This is a seperate function to allow plugins to be updated
        dynamically by other parts of the application.
        """
        plugins = {}
        _plugin_files = glob.glob(os.path.join(self.dir_path, '[!_]*.pyc'))
        for f in glob.glob(os.path.join(self.dir_path, '[!_]*.py')):
            if not any(os.path.splitext(f)[0] == os.path.splitext(x)[0]
                    for x in _plugin_files):
                logging.debug('Adding plugin {0}'.format(f))
                _plugin_files.append(f)
        for f in _plugin_files:
            plugin = self._load_compiled(f)
            plugins[plugin.__name__] = plugin
            logging.debug('Loaded module object for plugin: {0}'.format(f))
        return plugins
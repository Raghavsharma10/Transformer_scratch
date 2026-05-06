def add_plugin(self, phase, name, args, reason=None):
        """
        if config has plugin, override it, else add it
        """
        plugin_modified = False

        for plugin in self.template[phase]:
            if plugin['name'] == name:
                plugin['args'] = args
                plugin_modified = True

        if not plugin_modified:
            self.template[phase].append({"name": name, "args": args})
            if reason:
                logger.info('{}:{} with args {}, {}'.format(phase, name, args, reason))
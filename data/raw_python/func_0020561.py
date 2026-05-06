def add_plugin(self, plugin_type, plugin_name, args_dict):
        """
        if config has plugin, override it, else add it
        """

        plugin_modified = False

        for plugin in self.dock_json[plugin_type]:
            if plugin['name'] == plugin_name:
                plugin['args'] = args_dict
                plugin_modified = True

        if not plugin_modified:
            self.dock_json[plugin_type].append({"name": plugin_name, "args": args_dict})
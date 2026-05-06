def remove_plugin(self, plugin_type, plugin_name):
        """
        if config contains plugin, remove it
        """
        for p in self.dock_json[plugin_type]:
            if p.get('name') == plugin_name:
                self.dock_json[plugin_type].remove(p)
                break
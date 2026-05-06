def get_all_plugins(self):
        """
        Gets all loaded plugins

        :return: List of all plugins
        """
        return [{
            "manifest": i,
            "plugin": self.get_plugin(i["name"]),
            "module": self.get_module(i["name"])
        } for i in self._manifests]
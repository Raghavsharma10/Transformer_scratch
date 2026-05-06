def reload_lv2_plugins_data(self):
        """
        Search for LV2 audio plugins in the system and extract the metadata
        needed by pluginsmanager to generate audio plugins.
        """
        plugins_data = self.lv2_builder.lv2_plugins_data()
        self._dao.save(plugins_data)
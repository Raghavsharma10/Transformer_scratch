def load_save_state(self) -> SaveState:
        """
        Load the savestate of the plugin.

        :return: savestate
        :rtype: ~unidown.plugin.save_state.SaveState
        :raises ~unidown.plugin.exceptions.PluginException: broken savestate json
        :raises ~unidown.plugin.exceptions.PluginException: different savestate versions
        :raises ~unidown.plugin.exceptions.PluginException: different plugin versions
        :raises ~unidown.plugin.exceptions.PluginException: different plugin names
        :raises ~unidown.plugin.exceptions.PluginException: could not parse the protobuf
        """
        if not self._save_state_file.exists():
            self.log.info("No savestate file found.")
            return SaveState(dynamic_data.SAVE_STATE_VERSION, self.info, datetime(1970, 1, 1), {})

        savestat_proto = ""
        with self._save_state_file.open(mode='r', encoding="utf8") as data_file:
            try:
                savestat_proto = json_format.Parse(data_file.read(), SaveStateProto(), ignore_unknown_fields=False)
            except ParseError:
                raise PluginException(
                    f"Broken savestate json. Please fix or delete (you may lose data in this case) the file: {self._save_state_file}")

        try:
            save_state = SaveState.from_protobuf(savestat_proto)
        except ValueError as ex:
            raise PluginException(f"Could not parse the protobuf {self._save_state_file}: {ex}")
        else:
            del savestat_proto

        if save_state.version != dynamic_data.SAVE_STATE_VERSION:
            raise PluginException("Different save state version handling is not implemented yet.")

        if save_state.plugin_info.version != self.info.version:
            raise PluginException("Different plugin version handling is not implemented yet.")

        if save_state.plugin_info.name != self.name:
            raise PluginException("Save state plugin ({name}) does not match the current ({cur_name}).".format(
                name=save_state.plugin_info.name, cur_name=self.name))
        return save_state
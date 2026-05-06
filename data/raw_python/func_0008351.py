def save_save_state(self, data_dict: Dict[str, LinkItem]):  # TODO: add progressbar
        """
        Save meta data about the downloaded things and the plugin to file.

        :param data_dict: data
        :type data_dict: Dict[link, ~unidown.plugin.link_item.LinkItem]
        """
        json_data = json_format.MessageToJson(self._create_save_state(data_dict).to_protobuf())
        with self._save_state_file.open(mode='w', encoding="utf8") as writer:
            writer.write(json_data)
def _create_save_state(self, link_item_dict: Dict[str, LinkItem]) -> SaveState:
        """
        Create protobuf savestate of the module and the given data.

        :param link_item_dict: data
        :type link_item_dict: Dict[str, ~unidown.plugin.link_item.LinkItem]
        :return: the savestate
        :rtype: ~unidown.plugin.save_state.SaveState
        """
        return SaveState(dynamic_data.SAVE_STATE_VERSION, self.info, self.last_update, link_item_dict)
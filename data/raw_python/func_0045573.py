def _make_instance(self, node_data):
        """
        Create a ListNode instance from node_data

        Args:
            node_data (dict): Data to create ListNode item.
        Returns:
            ListNode item.
        """
        node_data['from_db'] = self._from_db
        clone = self.__call__(**node_data)
        clone.setattrs(container = self,
                    _is_item = True)
        for name in self._nodes:
            _name = un_camel(name)
            if _name in node_data:  # check for partial data
                getattr(clone, name)._load_data(node_data[_name])
        _key = clone._get_linked_model_key()
        if _key:
            self.node_dict[_key] = clone
        return clone
def get_data_item_for_reference_key(self, data_item_reference_key: str=None, create_if_needed: bool=False, large_format: bool=False) -> DataItem:
        """Get the data item associated with data item reference key. Optionally create if missing.

        :param data_item_reference_key: The data item reference key.
        :param create_if_needed: Whether to create a new data item if none is found.
        :return: The associated data item. May be None.

        .. versionadded:: 1.0

        Status: Provisional
        Scriptable: Yes
        """
        document_model = self._document_model
        data_item_reference = document_model.get_data_item_reference(data_item_reference_key)
        data_item = data_item_reference.data_item
        if data_item is None and create_if_needed:
            data_item = DataItemModule.DataItem(large_format=large_format)
            data_item.ensure_data_source()
            document_model.append_data_item(data_item)
            document_model.setup_channel(data_item_reference_key, data_item)
            data_item.session_id = document_model.session_id
            data_item = document_model.get_data_item_reference(data_item_reference_key).data_item
        return DataItem(data_item) if data_item else None
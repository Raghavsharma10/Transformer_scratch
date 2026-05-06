def get_data_item_by_uuid(self, data_item_uuid: uuid_module.UUID) -> DataItem:
        """Get the data item with the given UUID.

        .. versionadded:: 1.0

        Status: Provisional
        Scriptable: Yes
        """
        data_item = self._document_model.get_data_item_by_uuid(data_item_uuid)
        return DataItem(data_item) if data_item else None
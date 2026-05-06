def snapshot_data_item(self, data_item: DataItem) -> DataItem:
        """Snapshot a data item. Similar to copy but with a data snapshot.

        .. versionadded:: 1.0

        Scriptable: No
        """
        data_item = data_item._data_item.snapshot()
        self.__document_model.append_data_item(data_item)
        return DataItem(data_item)
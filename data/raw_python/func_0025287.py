def copy_data_item(self, data_item: DataItem) -> DataItem:
        """Copy a data item.

        .. versionadded:: 1.0

        Scriptable: No
        """
        data_item = copy.deepcopy(data_item._data_item)
        self.__document_model.append_data_item(data_item)
        return DataItem(data_item)
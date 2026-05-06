def get_data_item_for_reference_key(self, data_item_reference_key: str=None, create_if_needed: bool=False, large_format: bool=False) -> DataItem:
        """Get the data item associated with data item reference key. Optionally create if missing.

        :param data_item_reference_key: The data item reference key.
        :param create_if_needed: Whether to create a new data item if none is found.
        :return: The associated data item. May be None.

        .. versionadded:: 1.0

        Status: Provisional
        Scriptable: Yes
        """
        ...
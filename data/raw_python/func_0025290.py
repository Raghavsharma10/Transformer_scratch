def get_data_item_for_hardware_source(self, hardware_source, channel_id: str=None, processor_id: str=None, create_if_needed: bool=False, large_format: bool=False) -> DataItem:
        """Get the data item associated with hardware source and (optional) channel id and processor_id. Optionally create if missing.

        :param hardware_source: The hardware_source.
        :param channel_id: The (optional) channel id.
        :param processor_id: The (optional) processor id for the channel.
        :param create_if_needed: Whether to create a new data item if none is found.
        :return: The associated data item. May be None.

        .. versionadded:: 1.0

        Status: Provisional
        Scriptable: Yes
        """
        assert hardware_source is not None
        hardware_source_id = hardware_source._hardware_source.hardware_source_id
        document_model = self._document_model
        data_item_reference_key = document_model.make_data_item_reference_key(hardware_source_id, channel_id, processor_id)
        return self.get_data_item_for_reference_key(data_item_reference_key, create_if_needed=create_if_needed, large_format=large_format)
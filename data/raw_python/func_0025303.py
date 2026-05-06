def create_data_item_from_data_and_metadata(self, data_and_metadata: DataAndMetadata.DataAndMetadata, title: str=None) -> DataItem:
        """Create a data item in the library from the data and metadata.

        .. versionadded:: 1.0
        .. deprecated:: 1.1
           Use :py:meth:`~nion.swift.Facade.Library.create_data_item_from_data_and_metadata` instead.

        Scriptable: No
        """
        data_item = DataItemModule.new_data_item(data_and_metadata)
        if title is not None:
            data_item.title = title
        self.__document_controller.document_model.append_data_item(data_item)
        return DataItem(data_item)
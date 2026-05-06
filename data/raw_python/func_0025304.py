def get_or_create_data_group(self, title: str) -> DataGroup:
        """Get (or create) a data group.

        .. versionadded:: 1.0
        .. deprecated:: 1.1
           Use :py:meth:`~nion.swift.Facade.Library.create_data_item_from_data` instead.

        Scriptable: No
        """
        return DataGroup(self.__document_controller.document_model.get_or_create_data_group(title))
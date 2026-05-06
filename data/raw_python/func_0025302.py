def create_data_item_from_data(self, data: numpy.ndarray, title: str=None) -> DataItem:
        """Create a data item in the library from data.

        .. versionadded:: 1.0
        .. deprecated:: 1.1
           Use :py:meth:`~nion.swift.Facade.Library.create_data_item_from_data` instead.

        Scriptable: No
        """
        return DataItem(self.__document_controller.add_data(data, title))
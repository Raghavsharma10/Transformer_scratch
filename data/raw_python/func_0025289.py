def get_or_create_data_group(self, title: str) -> DataGroup:
        """Get (or create) a data group.

        :param title: The title of the data group.
        :return: The new :py:class:`nion.swift.Facade.DataGroup` object.
        :rtype: :py:class:`nion.swift.Facade.DataGroup`

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        return DataGroup(self.__document_model.get_or_create_data_group(title))
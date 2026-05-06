def display_items(self) -> typing.List[Display]:
        """Return the list of display items.

        :return: The list of :py:class:`nion.swift.Facade.Display` objects.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        return [Display(display_item) for display_item in self.__document_model.display_items]
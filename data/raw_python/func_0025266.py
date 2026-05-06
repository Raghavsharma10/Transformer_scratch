def graphics(self) -> typing.List[Graphic]:
        """Return the graphics attached to this data item.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        return [Graphic(graphic) for graphic in self.__display_item.graphics]
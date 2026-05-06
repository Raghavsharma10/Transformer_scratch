def get_graphic_by_uuid(self, graphic_uuid: uuid_module.UUID) -> Graphic:
        """Get the graphic with the given UUID.

        .. versionadded:: 1.0

        Status: Provisional
        Scriptable: Yes
        """
        for display_item in self._document_model.display_items:
            for graphic in display_item.graphics:
                if graphic.uuid == graphic_uuid:
                    return Graphic(graphic)
        return None
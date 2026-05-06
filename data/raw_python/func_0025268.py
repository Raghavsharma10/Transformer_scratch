def mask_xdata(self) -> DataAndMetadata.DataAndMetadata:
        """Return the mask by combining any mask graphics on this data item as extended data.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        display_data_channel = self.__display_item.display_data_channel
        shape = display_data_channel.display_data_shape
        mask = numpy.zeros(shape)
        for graphic in self.__display_item.graphics:
            if isinstance(graphic, (Graphics.SpotGraphic, Graphics.WedgeGraphic, Graphics.RingGraphic, Graphics.LatticeGraphic)):
                mask = numpy.logical_or(mask, graphic.get_mask(shape))
        return DataAndMetadata.DataAndMetadata.from_data(mask)
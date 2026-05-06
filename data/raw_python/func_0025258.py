def mask_xdata_with_shape(self, shape: DataAndMetadata.ShapeType) -> DataAndMetadata.DataAndMetadata:
        """Return the mask created by this graphic as extended data.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        mask = self._graphic.get_mask(shape)
        return DataAndMetadata.DataAndMetadata.from_data(mask)
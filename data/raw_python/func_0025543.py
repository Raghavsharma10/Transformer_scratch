def create_data_and_metadata(self, data: numpy.ndarray, intensity_calibration: Calibration.Calibration=None, dimensional_calibrations: typing.List[Calibration.Calibration]=None, metadata: dict=None, timestamp: str=None, data_descriptor: DataAndMetadata.DataDescriptor=None) -> DataAndMetadata.DataAndMetadata:
        """Create a data_and_metadata object from data.

        :param data: an ndarray of data.
        :param intensity_calibration: An optional calibration object.
        :param dimensional_calibrations: An optional list of calibration objects.
        :param metadata: A dict of metadata.
        :param timestamp: A datetime object.
        :param data_descriptor: A data descriptor describing the dimensions.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        ...
def create_data_and_metadata_from_data(self, data: numpy.ndarray, intensity_calibration: Calibration.Calibration=None, dimensional_calibrations: typing.List[Calibration.Calibration]=None, metadata: dict=None, timestamp: str=None) -> DataAndMetadata.DataAndMetadata:
        """Create a data_and_metadata object from data.

        .. versionadded:: 1.0
        .. deprecated:: 1.1
           Use :py:meth:`~nion.swift.Facade.DataItem.create_data_and_metadata` instead.

        Scriptable: No
        """
        ...
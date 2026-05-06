def set_dimensional_calibrations(self, dimensional_calibrations: typing.List[CalibrationModule.Calibration]) -> None:
        """Set the dimensional calibrations.

        :param dimensional_calibrations: A list of calibrations, must match the dimensions of the data.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        self.__data_item.set_dimensional_calibrations(dimensional_calibrations)
def create_calibration(self, offset: float=None, scale: float=None, units: str=None) -> Calibration.Calibration:
        """Create a calibration object with offset, scale, and units.

        :param offset: The offset of the calibration.
        :param scale: The scale of the calibration.
        :param units: The units of the calibration as a string.
        :return: The calibration object.

        .. versionadded:: 1.0

        Scriptable: Yes

        Calibrated units and uncalibrated units have the following relationship:
            :samp:`calibrated_value = offset + value * scale`
        """
        ...
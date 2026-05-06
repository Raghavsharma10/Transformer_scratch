def set_units(self, unit):
        """Set the unit for this data point

        Unit, as with data_type, are actually associated with the stream and not
        the individual data point.  As such, changing this within a stream is
        not encouraged.  Setting the unit on the data point is useful when the
        stream might be created with the write of a data point.

        """
        self._units = validate_type(unit, type(None), *six.string_types)
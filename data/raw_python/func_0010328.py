def get_data_type(self, use_cached=True):
        """Get the data type of this stream if it exists

        The data type is the type of data stored in this data stream. Valid types include:

        * INTEGER - data can be represented with a network (= big-endian) 32-bit two's-complement integer.  Data
          with this type maps to a python int.
        * LONG - data can be represented with a network (= big-endian) 64-bit two's complement integer.  Data
          with this type maps to a python int.
        * FLOAT - data can be represented with a network (= big-endian) 32-bit IEEE754 floating point.  Data
          with this type maps to a python float.
        * DOUBLE - data can be represented with a network (= big-endian) 64-bit IEEE754 floating point.  Data
          with this type maps to a python float.
        * STRING - UTF-8.  Data with this type map to a python string
        * BINARY - Data with this type map to a python string.
        * UNKNOWN - Data with this type map to a python string.

        :param bool use_cached: If False, the function will always request the latest from Device Cloud.
            If True, the device will not make a request if it already has cached data.
        :return: The data type of this stream as a string
        :rtype: str

        """
        dtype = self._get_stream_metadata(use_cached).get("dataType")
        if dtype is not None:
            dtype = dtype.upper()
        return dtype
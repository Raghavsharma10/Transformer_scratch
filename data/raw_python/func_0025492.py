def set_data_and_metadata(self, data_and_metadata, data_modified=None):
        """Sets the underlying data and data-metadata to the data_and_metadata.

        Note: this does not make a copy of the data.
        """
        self.increment_data_ref_count()
        try:
            if data_and_metadata:
                data = data_and_metadata.data
                data_shape_and_dtype = data_and_metadata.data_shape_and_dtype
                intensity_calibration = data_and_metadata.intensity_calibration
                dimensional_calibrations = data_and_metadata.dimensional_calibrations
                metadata = data_and_metadata.metadata
                timestamp = data_and_metadata.timestamp
                data_descriptor = data_and_metadata.data_descriptor
                timezone = data_and_metadata.timezone or Utility.get_local_timezone()
                timezone_offset = data_and_metadata.timezone_offset or Utility.TimezoneMinutesToStringConverter().convert(Utility.local_utcoffset_minutes())
                new_data_and_metadata = DataAndMetadata.DataAndMetadata(self.__load_data, data_shape_and_dtype, intensity_calibration, dimensional_calibrations, metadata, timestamp, data, data_descriptor, timezone, timezone_offset)
            else:
                new_data_and_metadata = None
            self.__set_data_metadata_direct(new_data_and_metadata, data_modified)
            if self.__data_and_metadata is not None:
                if self.persistent_object_context and not self.persistent_object_context.is_write_delayed(self):
                    self.persistent_object_context.write_external_data(self, "data", self.__data_and_metadata.data)
                    self.__data_and_metadata.unloadable = True
        finally:
            self.decrement_data_ref_count()
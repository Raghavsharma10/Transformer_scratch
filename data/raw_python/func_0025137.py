def update(self, data_and_metadata: DataAndMetadata.DataAndMetadata, state: str, sub_area, view_id) -> None:
        """Called from hardware source when new data arrives."""
        self.__state = state
        self.__sub_area = sub_area

        hardware_source_id = self.__hardware_source.hardware_source_id
        channel_index = self.index
        channel_id = self.channel_id
        channel_name = self.name
        metadata = copy.deepcopy(data_and_metadata.metadata)
        hardware_source_metadata = dict()
        hardware_source_metadata["hardware_source_id"] = hardware_source_id
        hardware_source_metadata["channel_index"] = channel_index
        if channel_id is not None:
            hardware_source_metadata["reference_key"] = "_".join([hardware_source_id, channel_id])
            hardware_source_metadata["channel_id"] = channel_id
        else:
            hardware_source_metadata["reference_key"] = hardware_source_id
        if channel_name is not None:
            hardware_source_metadata["channel_name"] = channel_name
        if view_id:
            hardware_source_metadata["view_id"] = view_id
        metadata.setdefault("hardware_source", dict()).update(hardware_source_metadata)

        data = data_and_metadata.data
        master_data = self.__data_and_metadata.data if self.__data_and_metadata else None
        data_matches = master_data is not None and data.shape == master_data.shape and data.dtype == master_data.dtype
        if data_matches and sub_area is not None:
            top = sub_area[0][0]
            bottom = sub_area[0][0] + sub_area[1][0]
            left = sub_area[0][1]
            right = sub_area[0][1] + sub_area[1][1]
            if top > 0 or left > 0 or bottom < data.shape[0] or right < data.shape[1]:
                master_data = numpy.copy(master_data)
                master_data[top:bottom, left:right] = data[top:bottom, left:right]
            else:
                master_data = numpy.copy(data)
        else:
            master_data = data  # numpy.copy(data). assume data does not need a copy.

        data_descriptor = data_and_metadata.data_descriptor
        intensity_calibration = data_and_metadata.intensity_calibration if data_and_metadata else None
        dimensional_calibrations = data_and_metadata.dimensional_calibrations if data_and_metadata else None
        timestamp = data_and_metadata.timestamp
        new_extended_data = DataAndMetadata.new_data_and_metadata(master_data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, metadata=metadata, timestamp=timestamp, data_descriptor=data_descriptor)

        self.__data_and_metadata = new_extended_data

        self.data_channel_updated_event.fire(new_extended_data)
        self.is_dirty = True
def __view_to_intervals(self, data_and_metadata: DataAndMetadata.DataAndMetadata, intervals: typing.List[typing.Tuple[float, float]]) -> None:
        """Change the view to encompass the channels and data represented by the given intervals."""
        left = None
        right = None
        for interval in intervals:
            left = min(left, interval[0]) if left is not None else interval[0]
            right = max(right, interval[1]) if right is not None else interval[1]
        left = left if left is not None else 0.0
        right = right if right is not None else 1.0
        left_channel = int(max(0.0, left) * data_and_metadata.data_shape[-1])
        right_channel = int(min(1.0, right) * data_and_metadata.data_shape[-1])
        data_min = numpy.amin(data_and_metadata.data[..., left_channel:right_channel])
        data_max = numpy.amax(data_and_metadata.data[..., left_channel:right_channel])
        if data_min > 0 and data_max > 0:
            y_min = 0.0
            y_max = data_max * 1.2
        elif data_min < 0 and data_max < 0:
            y_min = data_min * 1.2
            y_max = 0.0
        else:
            y_min = data_min * 1.2
            y_max = data_max * 1.2
        extra = (right - left) * 0.5
        display_left_channel = int(max(0.0, left - extra) * data_and_metadata.data_shape[-1])
        display_right_channel = int(min(1.0, right + extra) * data_and_metadata.data_shape[-1])
        # command = self.delegate.create_change_display_command()
        self.delegate.update_display_properties({"left_channel": display_left_channel, "right_channel": display_right_channel, "y_min": y_min, "y_max": y_max})
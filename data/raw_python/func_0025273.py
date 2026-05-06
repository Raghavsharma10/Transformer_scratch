def record(self, frame_parameters: dict=None, channels_enabled: typing.List[bool]=None, timeout: float=None) -> typing.List[DataAndMetadata.DataAndMetadata]:
        """Record data and return a list of data_and_metadata objects.

        .. versionadded:: 1.0

        :param frame_parameters: The frame parameters for the record. Pass None for defaults.
        :type frame_parameters: :py:class:`FrameParameters`
        :param channels_enabled: The enabled channels for the record. Pass None for defaults.
        :type channels_enabled: List of booleans.
        :param timeout: The timeout in seconds. Pass None to use default.
        :return: The list of data and metadata items that were read.
        :rtype: list of :py:class:`DataAndMetadata`
        """
        if frame_parameters:
            self.__hardware_source.set_record_frame_parameters(self.__hardware_source.get_frame_parameters_from_dict(frame_parameters))
        if channels_enabled is not None:
            for channel_index, channel_enabled in enumerate(channels_enabled):
                self.__hardware_source.set_channel_enabled(channel_index, channel_enabled)
        self.__hardware_source.start_recording()
        return self.__hardware_source.get_next_xdatas_to_finish(timeout)
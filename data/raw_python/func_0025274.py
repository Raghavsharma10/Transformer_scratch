def create_record_task(self, frame_parameters: dict=None, channels_enabled: typing.List[bool]=None) -> RecordTask:
        """Create a record task for this hardware source.

        .. versionadded:: 1.0

        :param frame_parameters: The frame parameters for the record. Pass None for defaults.
        :type frame_parameters: :py:class:`FrameParameters`
        :param channels_enabled: The enabled channels for the record. Pass None for defaults.
        :type channels_enabled: List of booleans.
        :return: The :py:class:`RecordTask` object.
        :rtype: :py:class:`RecordTask`

        Callers should call close on the returned task when finished.

        See :py:class:`RecordTask` for examples of how to use.
        """
        return RecordTask(self.__hardware_source, frame_parameters, channels_enabled)
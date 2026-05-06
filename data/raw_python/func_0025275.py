def create_view_task(self, frame_parameters: dict=None, channels_enabled: typing.List[bool]=None, buffer_size: int=1) -> ViewTask:
        """Create a view task for this hardware source.

        .. versionadded:: 1.0

        :param frame_parameters: The frame parameters for the view. Pass None for defaults.
        :type frame_parameters: :py:class:`FrameParameters`
        :param channels_enabled: The enabled channels for the view. Pass None for defaults.
        :type channels_enabled: List of booleans.
        :param buffer_size: The buffer size if using the grab_earliest method. Default is 1.
        :type buffer_size: int
        :return: The :py:class:`ViewTask` object.
        :rtype: :py:class:`ViewTask`

        Callers should call close on the returned task when finished.

        See :py:class:`ViewTask` for examples of how to use.
        """
        return ViewTask(self.__hardware_source, frame_parameters, channels_enabled, buffer_size)
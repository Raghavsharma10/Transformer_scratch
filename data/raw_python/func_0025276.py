def grab_next_to_finish(self, timeout: float=None) -> typing.List[DataAndMetadata.DataAndMetadata]:
        """Grabs the next frame to finish and returns it as data and metadata.

        .. versionadded:: 1.0

        :param timeout: The timeout in seconds. Pass None to use default.
        :return: The list of data and metadata items that were read.
        :rtype: list of :py:class:`DataAndMetadata`

        If the view is not already started, it will be started automatically.

        Scriptable: Yes
        """
        self.start_playing()
        return self.__hardware_source.get_next_xdatas_to_finish(timeout)
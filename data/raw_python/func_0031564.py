def get_flash_region(self, offset, length):
        """
        Retrieves the contents of a region of flash from the watch. This only works on watches running
        non-release firmware.
        Raises :exc:`.GetBytesError` on failure.

        :return: The retrieved data
        :rtype: bytes
        """
        return self._get(GetBytesFlashRequest(offset=offset, length=length))
def grab_image(self):
        """
        Takes a screenshot. Blocks until completion, or raises a :exc:`.ScreenshotError` on failure.

        While this method is executing, "progress" events will periodically be emitted with the following signature: ::

           (downloaded_so_far, total_size)

        :return: A list of bytearrays in RGB8 format, where each bytearray is one row of the image.
        """
        # We have to open this queue before we make the request, to ensure we don't miss the response.
        queue = self._pebble.get_endpoint_queue(ScreenshotResponse)
        self._pebble.send_packet(ScreenshotRequest())
        return self._read_screenshot(queue)
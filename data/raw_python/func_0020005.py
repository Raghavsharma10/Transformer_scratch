def determine_size(self, capture):
        """Determines the height and width of the image source.

        If no dimensions are available, this method defaults to a resolution of
        640x480, thus returns (480, 640).
        If capture has a get method it is assumed to understand
        `cv2.CAP_PROP_FRAME_WIDTH` and `cv2.CAP_PROP_FRAME_HEIGHT` to get the
        information. Otherwise it reads one frame from the source to determine
        image dimensions.

        Args:
            capture: the source to read from.

        Returns:
            A tuple containing integers of height and width (simple casts).
        """
        width = 640
        height = 480
        if capture and hasattr(capture, 'get'):
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        else:
            self.frame_offset += 1
            ret, frame = capture.read()
            if ret:
                width = frame.shape[1]
                height = frame.shape[0]
        return (int(height), int(width))
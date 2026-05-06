def read_frame(self):
        """Reads a frame and converts the color if needed.

        In case no frame is available, i.e. self.capture.read() returns False
        as the first return value, the event_source of the TimedAnimation is
        stopped, and if possible the capture source released.

        Returns:
            None if stopped, otherwise the color converted source image.
        """
        ret, frame = self.capture.read()
        if not ret:
            self.event_source.stop()
            try:
                self.capture.release()
            except AttributeError:
                # has no release method, thus just pass
                pass
            return None
        if self.convert_color != -1 and is_color_image(frame):
            return cv2.cvtColor(frame, self.convert_color)
        return frame
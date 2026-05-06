def frame_received_cb(self, frame):
        """Received message."""
        PYVLXLOG.debug("REC: %s", frame)
        for frame_received_cb in self.frame_received_cbs:
            # pylint: disable=not-callable
            self.loop.create_task(frame_received_cb(frame))
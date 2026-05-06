def write(self, frame):
        """Write frame to Bus."""
        if not isinstance(frame, FrameBase):
            raise PyVLXException("Frame not of type FrameBase", frame_type=type(frame))
        PYVLXLOG.debug("SEND: %s", frame)
        self.transport.write(slip_pack(bytes(frame)))
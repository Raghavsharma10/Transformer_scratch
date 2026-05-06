def validate_payload_len(self, payload):
        """Validate payload len."""
        if not hasattr(self, "PAYLOAD_LEN"):
            # No fixed payload len, e.g. within FrameGetSceneListNotification
            return
        # pylint: disable=no-member
        if len(payload) != self.PAYLOAD_LEN:
            raise PyVLXException("Invalid payload len", expected_len=self.PAYLOAD_LEN, current_len=len(payload), frame_type=type(self).__name__)
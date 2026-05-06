async def handle_frame(self, frame):
        """Handle incoming API frame, return True if this was the expected frame."""
        if not isinstance(frame, FrameGetStateConfirmation):
            return False
        self.success = True
        self.gateway_state = frame.gateway_state
        self.gateway_sub_state = frame.gateway_sub_state
        return True
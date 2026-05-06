async def handle_frame(self, frame):
        """Handle incoming API frame, return True if this was the expected frame."""
        if not isinstance(frame, FrameGetVersionConfirmation):
            return False
        self.version = frame.version
        self.success = True
        return True
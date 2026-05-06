async def handle_frame(self, frame):
        """Handle incoming API frame, return True if this was the expected frame."""
        if not isinstance(frame, FrameSetNodeNameConfirmation):
            return False
        self.success = frame.status == SetNodeNameConfirmationStatus.OK
        return True
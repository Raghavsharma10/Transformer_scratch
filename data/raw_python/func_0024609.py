async def handle_frame(self, frame):
        """Handle incoming API frame, return True if this was the expected frame."""
        if not isinstance(frame, FrameHouseStatusMonitorDisableConfirmation):
            return False
        self.success = True
        return True
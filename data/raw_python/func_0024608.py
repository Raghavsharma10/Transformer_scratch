async def handle_frame(self, frame):
        """Handle incoming API frame, return True if this was the expected frame."""
        if not isinstance(frame, FrameHouseStatusMonitorEnableConfirmation):
            return False
        self.success = True
        return True
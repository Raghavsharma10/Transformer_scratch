async def handle_frame(self, frame):
        """Handle incoming API frame, return True if this was the expected frame."""
        if not isinstance(frame, FramePasswordEnterConfirmation):
            return False
        if frame.status == PasswordEnterConfirmationStatus.FAILED:
            PYVLXLOG.warning('Failed to authenticate with password "%s****"', self.password[:2])
            self.success = False
        if frame.status == PasswordEnterConfirmationStatus.SUCCESSFUL:
            self.success = True
        return True
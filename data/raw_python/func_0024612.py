async def handle_frame(self, frame):
        """Handle incoming API frame, return True if this was the expected frame."""
        if isinstance(frame, FrameCommandSendConfirmation) and frame.session_id == self.session_id:
            if frame.status == CommandSendConfirmationStatus.ACCEPTED:
                self.success = True
            return not self.wait_for_completion
        if isinstance(frame, FrameCommandRemainingTimeNotification) and frame.session_id == self.session_id:
            # Ignoring FrameCommandRemainingTimeNotification
            return False
        if isinstance(frame, FrameCommandRunStatusNotification) and frame.session_id == self.session_id:
            # At the moment I don't reall understand what the FrameCommandRunStatusNotification is good for.
            # Ignoring these packets for now
            return False
        if isinstance(frame, FrameSessionFinishedNotification) and frame.session_id == self.session_id:
            return True
        return False
async def handle_frame(self, frame):
        """Handle incoming API frame, return True if this was the expected frame."""
        if isinstance(frame, FrameGetNodeInformationConfirmation) and frame.node_id == self.node_id:
            # We are still waiting for GetNodeInformationNotification
            return False
        if isinstance(frame, FrameGetNodeInformationNotification) and frame.node_id == self.node_id:
            self.notification_frame = frame
            self.success = True
            return True
        return False
def request_frame(self):
        """Construct initiating frame."""
        self.session_id = get_new_session_id()
        return FrameCommandSendRequest(node_ids=[self.node_id], parameter=self.parameter, session_id=self.session_id)
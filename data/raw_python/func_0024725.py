def request_frame(self):
        """Construct initiating frame."""
        self.session_id = get_new_session_id()
        return FrameActivateSceneRequest(scene_id=self.scene_id, session_id=self.session_id)
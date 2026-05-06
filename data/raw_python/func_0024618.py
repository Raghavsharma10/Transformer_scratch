async def handle_frame(self, frame):
        """Handle incoming API frame, return True if this was the expected frame."""
        if isinstance(frame, FrameGetSceneListConfirmation):
            self.count_scenes = frame.count_scenes
            if self.count_scenes == 0:
                self.success = True
                return True
            # We are still waiting for FrameGetSceneListNotification(s)
            return False
        if isinstance(frame, FrameGetSceneListNotification):
            self.scenes.extend(frame.scenes)
            if frame.remaining_scenes != 0:
                # We are still waiting for FrameGetSceneListConfirmation(s)
                return False
            if self.count_scenes != len(self.scenes):
                PYVLXLOG.warning("Warning: number of received scenes does not match expected number")
            self.success = True
            return True
        return False
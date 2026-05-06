def _update_fps(self, event):
        """Update the fps after every window"""
        self._frame_count += 1
        diff = time() - self._basetime
        if (diff > self._fps_window):
            self._fps = self._frame_count / diff
            self._basetime = time()
            self._frame_count = 0
            self._fps_callback(self.fps)
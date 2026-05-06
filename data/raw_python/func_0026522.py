def _toggle_filming(self):
        """Toggles the camera system recording state"""

        if self._filming:
            self.log("Stopping operation")
            self._filming = False
            self.timer.stop()
        else:
            self.log("Starting operation")
            self._filming = True
            self.timer.start()
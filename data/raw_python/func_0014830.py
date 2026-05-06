def move_down(self):
        """Make the drone decent downwards."""
        self.at(ardrone.at.pcmd, True, 0, 0, -self.speed, 0)
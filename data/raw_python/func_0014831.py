def move_forward(self):
        """Make the drone move forward."""
        self.at(ardrone.at.pcmd, True, 0, -self.speed, 0, 0)
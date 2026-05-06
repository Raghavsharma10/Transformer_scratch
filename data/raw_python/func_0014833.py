def turn_left(self):
        """Make the drone rotate left."""
        self.at(ardrone.at.pcmd, True, 0, 0, 0, -self.speed)
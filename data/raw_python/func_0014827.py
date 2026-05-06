def move_left(self):
        """Make the drone move left."""
        self.at(ardrone.at.pcmd, True, -self.speed, 0, 0, 0)
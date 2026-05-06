def move_right(self):
        """Make the drone move right."""
        self.at(ardrone.at.pcmd, True, self.speed, 0, 0, 0)
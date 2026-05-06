def move_up(self):
        """Make the drone rise upwards."""
        self.at(ardrone.at.pcmd, True, 0, 0, self.speed, 0)
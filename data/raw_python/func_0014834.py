def turn_right(self):
        """Make the drone rotate right."""
        self.at(ardrone.at.pcmd, True, 0, 0, 0, self.speed)
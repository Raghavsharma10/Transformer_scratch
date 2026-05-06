def move_backward(self):
        """Make the drone move backwards."""
        self.at(ardrone.at.pcmd, True, 0, self.speed, 0, 0)
def reset(self):
        """Toggle the drone's emergency state."""
        self.at(ardrone.at.ref, False, True)
        time.sleep(0.1)
        self.at(ardrone.at.ref, False, False)
def sleep(self):
        """Send the controller to sleep"""
        logger.debug("Sleep the controller")
        self.write(Registers.MODE_1, self.mode_1 | (1 << Mode1.SLEEP))
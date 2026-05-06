def equals(self, controller):
        """ Verify if the controller corresponds
            to the current one.

        """
        if controller is None:
            return False

        return self.user == controller.user and self.enterprise == controller.enterprise and self.url == controller.url
def activate(self):
        """
        Activates the bounce instance and updates it with the latest data.

        :return: Activation status.
        :rtype: `str`
        """
        response = self._manager.activate(self.ID)
        self._update(response["Bounce"])
        return response["Message"]
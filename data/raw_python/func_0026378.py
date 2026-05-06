def on_pumprequest(self, event):
        """
        Activates or deactivates a connected pump.

        :param event:
        """
        self.log("Updating pump status: ", event.controlvalue)
        self._set_digital_pin(self._pump_channel, event.controlvalue)
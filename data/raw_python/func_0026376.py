def on_machinerequest(self, event):
        """
        Sets a new machine speed.

        :param event:
        """
        self.log("Updating new machine power: ", event.controlvalue)
        self._handle_servo(self._machine_channel, event.controlvalue)
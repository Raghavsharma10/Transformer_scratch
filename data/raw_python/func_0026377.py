def on_rudderrequest(self, event):
        """
        Sets a new rudder angle.

        :param event:
        """
        self.log("Updating new rudder angle: ", event.controlvalue)
        self._handle_servo(self._rudder_channel, event.controlvalue)
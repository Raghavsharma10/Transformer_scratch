def _disconnect(self, mqttc, userdata, rc):
        """
        The callback for when a DISCONNECT occurs.

        :param mqttc: The client instance for this callback
        :param userdata: The private userdata for the mqtt client. Not used in Polyglot
        :param rc: Result code of connection, 0 = Graceful, anything else is unclean
        """
        self.connected = False
        if rc != 0:
            LOGGER.info("MQTT Unexpected disconnection. Trying reconnect.")
            try:
                self._mqttc.reconnect()
            except Exception as ex:
                template = "An exception of type {0} occured. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                LOGGER.error("MQTT Connection error: " + message)
        else:
            LOGGER.info("MQTT Graceful disconnection.")
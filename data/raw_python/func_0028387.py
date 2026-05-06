def _startMqtt(self):
        """
        The client start method. Starts the thread for the MQTT Client
        and publishes the connected message.
        """
        LOGGER.info('Connecting to MQTT... {}:{}'.format(self._server, self._port))
        try:
            # self._mqttc.connect_async(str(self._server), int(self._port), 10)
            self._mqttc.connect_async('{}'.format(self._server), int(self._port), 10)
            self._mqttc.loop_forever()
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            LOGGER.error("MQTT Connection error: {}".format(message), exc_info=True)
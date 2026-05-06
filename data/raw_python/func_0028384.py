def _connect(self, mqttc, userdata, flags, rc):
        """
        The callback for when the client receives a CONNACK response from the server.
        Subscribing in on_connect() means that if we lose the connection and
        reconnect then subscriptions will be renewed.

        :param mqttc: The client instance for this callback
        :param userdata: The private userdata for the mqtt client. Not used in Polyglot
        :param flags: The flags set on the connection.
        :param rc: Result code of connection, 0 = Success, anything else is a failure
        """
        if rc == 0:
            self.connected = True
            results = []
            LOGGER.info("MQTT Connected with result code " + str(rc) + " (Success)")
            # result, mid = self._mqttc.subscribe(self.topicInput)
            results.append((self.topicInput, tuple(self._mqttc.subscribe(self.topicInput))))
            results.append((self.topicPolyglotConnection, tuple(self._mqttc.subscribe(self.topicPolyglotConnection))))
            for (topic, (result, mid)) in results:
                if result == 0:
                    LOGGER.info("MQTT Subscribing to topic: " + topic + " - " + " MID: " + str(mid) + " Result: " + str(result))
                else:
                    LOGGER.info("MQTT Subscription to " + topic + " failed. This is unusual. MID: " + str(mid) + " Result: " + str(result))
                    # If subscription fails, try to reconnect.
                    self._mqttc.reconnect()
            self._mqttc.publish(self.topicSelfConnection, json.dumps(
                {
                    'connected': True,
                    'node': self.profileNum
                }), retain=True)
            LOGGER.info('Sent Connected message to Polyglot')
        else:
            LOGGER.error("MQTT Failed to connect. Result code: " + str(rc))
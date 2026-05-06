def send(self, message):
        """
        Formatted Message to send to Polyglot. Connection messages are sent automatically from this module
        so this method is used to send commands to/from Polyglot and formats it for consumption
        """
        if not isinstance(message, dict) and self.connected:
            warnings.warn('payload not a dictionary')
            return False
        try:
            message['node'] = self.profileNum
            self._mqttc.publish(self.topicInput, json.dumps(message), retain=False)
        except TypeError as err:
            LOGGER.error('MQTT Send Error: {}'.format(err), exc_info=True)
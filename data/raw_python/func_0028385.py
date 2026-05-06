def _message(self, mqttc, userdata, msg):
        """
        The callback for when a PUBLISH message is received from the server.

        :param mqttc: The client instance for this callback
        :param userdata: The private userdata for the mqtt client. Not used in Polyglot
        :param flags: The flags set on the connection.
        :param msg: Dictionary of MQTT received message. Uses: msg.topic, msg.qos, msg.payload
        """
        try:
            inputCmds = ['query', 'command', 'result', 'status', 'shortPoll', 'longPoll', 'delete']
            parsed_msg = json.loads(msg.payload.decode('utf-8'))
            if 'node' in parsed_msg:
                if parsed_msg['node'] != 'polyglot':
                    return
                del parsed_msg['node']
                for key in parsed_msg:
                    # LOGGER.debug('MQTT Received Message: {}: {}'.format(msg.topic, parsed_msg))
                    if key == 'config':
                        self.inConfig(parsed_msg[key])
                    elif key == 'connected':
                        self.polyglotConnected = parsed_msg[key]
                    elif key == 'stop':
                        LOGGER.debug('Received stop from Polyglot... Shutting Down.')
                        self.stop()
                    elif key in inputCmds:
                        self.input(parsed_msg)
                    else:
                        LOGGER.error('Invalid command received in message from Polyglot: {}'.format(key))

        except (ValueError) as err:
            LOGGER.error('MQTT Received Payload Error: {}'.format(err), exc_info=True)
def start_blocking(self, run_event):
        """ Start the MQTT client, as a blocking method.

        :param run_event: a run event object provided by the thread handler.
        """
        topics = [("hermes/intent/#", 0), ("hermes/hotword/#", 0), ("hermes/asr/#", 0), ("hermes/nlu/#", 0),
                  ("snipsmanager/#", 0)]

        self.log_info("Connecting to {} on port {}".format(self.mqtt_hostname, str(self.mqtt_port)))

        retry = 0
        while True and run_event.is_set():
            try:
                self.log_info("Trying to connect to {}".format(self.mqtt_hostname))
                self.client.connect(self.mqtt_hostname, self.mqtt_port, 60)
                break
            except (socket_error, Exception) as e:
                self.log_info("MQTT error {}".format(e))
                time.sleep(5 + int(retry / 5))
                retry = retry + 1

        topics = [
            (MQTT_TOPIC_INTENT + '#', 0),
            (MQTT_TOPIC_HOTWORD + '#', 0),
            (MQTT_TOPIC_ASR + '#', 0),
            (MQTT_TOPIC_SNIPSFILE, 0),
            (MQTT_TOPIC_DIALOG_MANAGER + '#', 0),
            ("snipsmanager/#", 0)
        ]
        self.client.subscribe(topics)

        while run_event.is_set():
            try:
                self.client.loop()
            except AttributeError as e:
                self.log_info("Error in mqtt run loop {}".format(e))
                time.sleep(1)
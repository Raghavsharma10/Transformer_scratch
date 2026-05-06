def on_message(self, client, userdata, msg):
        """ Callback when the MQTT client received a new message.

        :param client: the MQTT client.
        :param userdata: unused.
        :param msg: the MQTT message.
        """
        if msg is None:
            return

        self.log_info("New message on topic {}".format(msg.topic))
        self.log_debug("Payload {}".format(msg.payload))

        if msg.payload is None or len(msg.payload) == 0:
            pass

        if msg.payload:
            payload = json.loads(msg.payload.decode('utf-8'))
            site_id = payload.get('siteId')
            session_id = payload.get('sessionId')

        if msg.topic is not None and msg.topic.startswith(MQTT_TOPIC_INTENT) and msg.payload:
            payload = json.loads(msg.payload.decode('utf-8'))
            intent = IntentParser.parse(payload, self.registry.intent_classes)
            self.log_debug("Parsed intent: {}".format(intent))
            if self.handle_intent is not None:
                if intent is not None:
                    self.log_debug("New intent: {}".format(str(intent.intentName)))
                self.handle_intent(intent, payload)
        elif msg.topic is not None and msg.topic == MQTT_TOPIC_HOTWORD + "toggleOn":
            self.state_handler.set_state(State.hotword_toggle_on)
        elif MQTT_TOPIC_HOTWORD_DETECTED_RE.match(msg.topic):
            if not self.first_hotword_detected:
                self.client.publish(
                    "hermes/feedback/sound/toggleOff", payload=None, qos=0, retain=False)
                self.first_hotword_detected = True
            self.state_handler.set_state(State.hotword_detected)
            if self.handle_start_listening is not None:
                self.handle_start_listening()
        elif msg.topic == MQTT_TOPIC_ASR + "startListening":
            self.state_handler.set_state(State.asr_start_listening)
        elif msg.topic == MQTT_TOPIC_ASR + "textCaptured":
            self.state_handler.set_state(State.asr_text_captured)
            if msg.payload is not None:
                self.log_debug("Text captured: {}".format(str(msg.payload)))
            if self.handle_done_listening is not None:
                self.handle_done_listening()
            payload = json.loads(msg.payload.decode('utf-8'))
            if payload['text'] == '':
                self.handle_intent(None, None)
        elif msg.topic is not None and msg.topic == "hermes/nlu/intentNotRecognized":
            self.handle_intent(None, None)
        elif msg.topic == "snipsmanager/setSnipsfile" and msg.payload:
            self.state_handler.set_state(State.asr_text_captured)
        elif msg.topic == MQTT_TOPIC_SESSION_STARTED:
            self.state_handler.set_state(State.session_started)
            if self.handlers_dialogue_events is not None:
                self.handlers_dialogue_events(self.DIALOGUE_EVENT_STARTED, session_id, site_id)
        elif msg.topic == MQTT_TOPIC_SESSION_ENDED:
            self.state_handler.set_state(State.session_ended)
            if self.handlers_dialogue_events is not None:
                self.handlers_dialogue_events(self.DIALOGUE_EVENT_ENDED, session_id, site_id)
        elif msg.topic == MQTT_TOPIC_SESSION_QUEUED:
            self.state_handler.set_state(State.session_queued)
            if self.handlers_dialogue_events is not None:
                self.handlers_dialogue_events(self.DIALOGUE_EVENT_QUEUED, session_id, site_id)
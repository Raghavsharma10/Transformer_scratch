def on_connect(self, client, userdata, flags, result_code):
        """ Callback when the MQTT client is connected.

        :param client: the client being connected.
        :param userdata: unused.
        :param flags: unused.
        :param result_code: result code.
        """
        self.log_info("Connected with result code {}".format(result_code))
        self.state_handler.set_state(State.welcome)
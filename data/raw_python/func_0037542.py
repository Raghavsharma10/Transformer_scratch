def on_disconnect(self, client, userdata, result_code):
        """ Callback when the MQTT client is disconnected. In this case,
            the server waits five seconds before trying to reconnected.

        :param client: the client being disconnected.
        :param userdata: unused.
        :param result_code: result code.
        """
        self.log_info("Disconnected with result code " + str(result_code))
        self.state_handler.set_state(State.goodbye)
        time.sleep(5)
        self.thread_handler.run(target=self.start_blocking)
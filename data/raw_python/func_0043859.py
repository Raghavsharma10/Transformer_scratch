def send_data(self, data, delay=COMMAND_DELAY):
        """ Send data to connected gateway """

        time_since_last_send = time.time() - self._last_send
        delay = max(0, delay - time_since_last_send)
        time.sleep(delay)  # Ensure minim recommended delay since last send

        for item in data:
            data = bytes.fromhex(str(item.zfill(2)))
            try:
                self.sock.send(data)
            except ConnectionResetError as msg:
                _LOGGER.error("Error trying to connect to Russound controller. "
                              "Check that no other device or system is using the port that "
                              "you are trying to connect to. Try resetting the bridge you are using to connect.")
                _LOGGER.error(msg)
        self._last_send = time.time()
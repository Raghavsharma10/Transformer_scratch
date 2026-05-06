def handle_message(self, connection, sender, target, message):
        """
        Handles a received message
        """
        parts = message.strip().split(' ', 2)
        if parts and parts[0].lower() == '!bot':
            try:
                command = parts[1].lower()
            except IndexError:
                self.safe_send(connection, target, "No command given")
                return

            try:
                payload = parts[2]
            except IndexError:
                payload = ""

            self.__pool.enqueue(self._handle_command,
                                connection, sender, target, command, payload)
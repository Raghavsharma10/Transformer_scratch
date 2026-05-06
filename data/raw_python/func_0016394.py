def send(self, command: str, **kwargs: Any) -> None:
        """
        Send a message to the server.

        .. code-block:: python

            client.send("nick", nick="weatherbot")
            client.send("privmsg", target="#python", message="Hello, World!")

        """
        packed_command = pack_command(command, **kwargs).strip()
        self.send_raw(packed_command)
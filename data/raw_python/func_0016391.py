async def connect(self) -> None:
        """Open a connection to the defined server."""
        def protocol_factory() -> Protocol:
            return Protocol(client=self)

        _, protocol = await self.loop.create_connection(
            protocol_factory,
            host=self.host,
            port=self.port,
            ssl=self.ssl
        )  # type: Tuple[Any, Any]
        if self.protocol:
            self.protocol.close()
        self.protocol = protocol
        # TODO: Delete the following code line. It is currently kept in order
        # to not break the current existing codebase. Removing it requires a
        # heavy change in the test codebase.
        protocol.client = self
        self.trigger("client_connect")
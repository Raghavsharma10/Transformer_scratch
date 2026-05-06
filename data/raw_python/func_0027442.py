def post(self, request: Request) -> None:
        """
        Dispatches a request over middleware. Returns when message put onto outgoing channel by producer,
        does not wait for response from a consuming application i.e. is fire-and-forget
        :param request: The request to dispatch
        :return: None
        """

        if self._producer is None:
            raise ConfigurationException("Command Processor requires a BrightsideProducer to post to a Broker")
        if self._message_mapper_registry is None:
            raise ConfigurationException("Command Processor requires a BrightsideMessage Mapper Registry to post to a Broker")

        message_mapper = self._message_mapper_registry.lookup(request)
        message = message_mapper(request)
        self._message_store.add(message)
        self._producer.send(message)
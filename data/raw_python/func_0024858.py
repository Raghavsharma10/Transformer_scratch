def _init_grpc_publisher(self):
        """
        initialize the grpc publisher, builds the stub and then starts the grpc manager
        :return: None
        """
        self._stub = EventHub_pb2_grpc.PublisherStub(channel=self._channel)
        self.grpc_manager = Eventhub.GrpcManager(stub_call=self._stub.send,
                                                 on_msg_callback=self._publisher_callback,
                                                 metadata=self._generate_publish_headers().items())
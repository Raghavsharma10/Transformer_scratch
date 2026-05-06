def connect_to_kafka(self, bootstrap_servers='127.0.0.1:9092',
                         auto_offset_reset='latest',
                         client_id='Robot',
                         **kwargs
                         ):
        """Connect to kafka
        - ``bootstrap_servers``: default 127.0.0.1:9092
        - ``client_id``: default: Robot
        """

        self.connect_consumer(
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset=auto_offset_reset,
            client_id=client_id,
            **kwargs
        )
        self.connect_producer(bootstrap_servers=bootstrap_servers, client_id=client_id)
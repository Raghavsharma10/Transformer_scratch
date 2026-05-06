def connect_consumer(
            self,
            bootstrap_servers='127.0.0.1:9092',
            client_id='Robot',
            group_id=None,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            **kwargs
    ):
        """Connect kafka consumer.
    
        Keyword Arguments:
        - ``bootstrap_servers``: 'host[:port]' string (or list of 'host[:port]'
            strings) that the consumer should contact to bootstrap initial
            cluster metadata. This does not have to be the full node list.
            It just needs to have at least one broker that will respond to a
            Metadata API Request. Default: `127.0.0.1:9092`.
        - ``client_id`` (str): a name for this client. This string is passed in
            each request to servers and can be used to identify specific
            server-side log entries that correspond to this client. Also
            submitted to GroupCoordinator for logging with respect to
            consumer group administration. Default: `Robot`.
        - ``group_id`` (str or None): name of the consumer group to join for dynamic
            partition assignment (if enabled), and to use for fetching and
            committing offsets. If None, auto-partition assignment (via
            group coordinator) and offset commits are disabled.
            Default: `None`.
        - ``auto_offset_reset`` (str): A policy for resetting offsets on
            OffsetOutOfRange errors: `earliest` will move to the oldest
            available message, `latest` will move to the most recent. Any
            other value will raise the exception. Default: `latest`.
        - ``enable_auto_commit`` (bool): If true the consumer's offset will be
            periodically committed in the background. Default: `True`.
            
        Note:
        Configuration parameters are described in more detail at
        http://kafka-python.readthedocs.io/en/master/apidoc/KafkaConsumer.html
        """

        self.consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset=auto_offset_reset,
            client_id=client_id,
            group_id=group_id,
            enable_auto_commit=enable_auto_commit,
            **kwargs
        )
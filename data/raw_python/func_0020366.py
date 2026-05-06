def connect_producer(self, bootstrap_servers='127.0.0.1:9092', client_id='Robot', **kwargs):
        """A Kafka client that publishes records to the Kafka cluster.
    
        Keyword Arguments:
        - ``bootstrap_servers``: 'host[:port]' string (or list of 'host[:port]'
          strings) that the producer should contact to bootstrap initial
          cluster metadata. This does not have to be the full node list.
          It just needs to have at least one broker that will respond to a
          Metadata API Request. Default to `localhost:9092`.
        - ``client_id`` (str): a name for this client. This string is passed in
        each request to servers and can be used to identify specific
        server-side log entries that correspond to this client.
        Default: `Robot`.

        Note:
        Configuration parameters are described in more detail at
        http://kafka-python.readthedocs.io/en/master/apidoc/KafkaProducer.html
        """
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers, client_id=client_id, **kwargs)
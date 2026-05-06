def send(self, topic, value=None, timeout=60, key=None, partition=None, timestamp_ms=None):
        """Publish a message to a topic.

        - ``topic`` (str): topic where the message will be published
        - ``value``: message value. Must be type bytes, or be serializable to bytes via configured value_serializer.
          If value is None, key is required and message acts as a `delete`.
        - ``timeout``
        - ``key``: a key to associate with the message. Can be used to determine which partition
          to send the message to. If partition is None (and producer's partitioner config is left as default),
          then messages with the same key will be delivered to the same partition (but if key is None,
          partition is chosen randomly). Must be type bytes, or be serializable to bytes via configured key_serializer.
        - ``partition`` (int): optionally specify a partition.
          If not set, the partition will be selected using the configured `partitioner`.
        - ``timestamp_ms`` (int): epoch milliseconds (from Jan 1 1970 UTC) to use as the message timestamp.
          Defaults to current time.
        """
        future = self.producer.send(topic, value=value, key=key, partition=partition, timestamp_ms=timestamp_ms)
        future.get(timeout=timeout)
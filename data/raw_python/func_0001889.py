def pub(self, topic=b'', embed_topic=False):
        """
        Returns a callable that can be used to transmit a message, with a given
        ``topic``, in a publisher-subscriber fashion. Note that the sender
        function has a ``print`` like signature, with an infinite number of
        arguments. Each one being a part of the complete message.

        By default, no topic will be included into published messages. Being up
        to developers to include the topic, at the beginning of the first part
        (i.e. frame) of every published message, so that subscribers are able
        to receive them. For a different behaviour, check the embed_topic
        argument.

        :param topic: the topic that will be published to (default=b'')
        :type topic: bytes
        :param embed_topic: set for the topic to be automatically sent as the
                            first part (i.e. frame) of every published message
                            (default=False)
        :type embed_topic bool
        :rtype: function
        """
        if not isinstance(topic, bytes):
            error = 'Topic must be bytes'
            log.error(error)
            raise TypeError(error)

        sock = self.__sock(zmq.PUB)
        return self.__send_function(sock, topic, embed_topic)
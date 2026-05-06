def sub(self, topics=(b'',)):
        """
        Returns an iterable that can be used to iterate over incoming messages,
        that were published with one of the topics specified in ``topics``. Note
        that the iterable returns as many parts as sent by subscribed publishers.

        :param topics: a list of topics to subscribe to (default=b'')
        :type topics: list of bytes
        :rtype: generator
        """
        sock = self.__sock(zmq.SUB)

        for topic in topics:
            if not isinstance(topic, bytes):
                error = 'Topics must be a list of bytes'
                log.error(error)
                raise TypeError(error)
            sock.setsockopt(zmq.SUBSCRIBE, topic)

        return self.__recv_generator(sock)
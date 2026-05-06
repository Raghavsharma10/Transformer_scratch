def pull(self):
        """
        Returns an iterable that can be used to iterate over incoming messages,
        that were pushed by a push socket. Note that the iterable returns as
        many parts as sent by pushers.

        :rtype: generator
        """
        sock = self.__sock(zmq.PULL)
        return self.__recv_generator(sock)
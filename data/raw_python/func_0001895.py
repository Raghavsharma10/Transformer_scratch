def pair(self):
        """
        Returns a callable and an iterable respectively. Those can be used to
        both transmit a message and/or iterate over incoming messages, that were
        sent by a pair socket. Note that the iterable returns as many parts as
        sent by a pair. Also, the sender function has a ``print`` like signature,
        with an infinite number of arguments. Each one being a part of the
        complete message.

        :rtype: (function, generator)
        """
        sock = self.__sock(zmq.PAIR)
        return self.__send_function(sock), self.__recv_generator(sock)
def reply(self):
        """
        Returns a callable and an iterable respectively. Those can be used to
        both transmit a message and/or iterate over incoming messages,
        that were requested by a request socket. Note that the iterable returns
        as many parts as sent by requesters. Also, the sender function has a
        ``print`` like signature, with an infinite number of arguments. Each one
        being a part of the complete message.

        :rtype: (function, generator)
        """
        sock = self.__sock(zmq.REP)
        return self.__send_function(sock), self.__recv_generator(sock)
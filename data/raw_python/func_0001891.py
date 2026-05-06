def push(self):
        """
        Returns a callable that can be used to transmit a message in a push-pull
        fashion. Note that the sender function has a ``print`` like signature,
        with an infinite number of arguments. Each one being a part of the
        complete message.

        :rtype: function
        """
        sock = self.__sock(zmq.PUSH)
        return self.__send_function(sock)
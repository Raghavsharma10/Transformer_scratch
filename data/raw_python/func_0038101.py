def _dispatch_message(self):
        """
        _dispatch_message
        """
        while True:
            message = self.req_queue.get()
            if message is None:
                _logger.debug("_dispatch_message thread is terminated")
                return

            if message._type != MessageType.EVENT:
                self.__dispatch_message(message)
            elif message._type == MessageType.EVENT:
                self.__dispatch_event_message(message)
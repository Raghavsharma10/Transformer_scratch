def send(self, message):
        """
        Sends a message (synchronous)

        :param message: Message to send
        :return: Message response(s)
        """
        future = self.post(message)
        future.join()
        return future.result
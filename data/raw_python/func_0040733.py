def close(self):
        """
        Disconnects from the server
        """
        # Disconnect with a fancy message, then close connection
        if self._connection is not None:
            self._connection.disconnect("Bot is quitting")
            self._connection.close()
            self._connection = None

        # Stop the client loop
        self.__stopped.set()

        if self.__thread is not None:
            try:
                self.__thread.join(5)
            except RuntimeError:
                pass
            self.__thread = None
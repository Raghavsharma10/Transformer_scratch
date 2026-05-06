def on_welcome(self, connection, event):
        """
        Server welcome: we're connected
        """
        # Start the pool
        self.__pool.start()

        logging.info("! Connected to server '%s': %s",
                     event.source, event.arguments[0])
        connection.join("#cohorte")
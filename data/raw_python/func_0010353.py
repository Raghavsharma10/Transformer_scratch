def create_session(self, callback, monitor_id):
        """
        Creates and Returns a PushSession instance based on the input monitor
        and callback.  When data is received, callback will be invoked.
        If neither monitor or monitor_id are specified, throws an Exception.

        :param callback: Callback function to call when PublishMessage
            messages are received. Expects 1 argument which will contain the
            payload of the pushed message.  Additionally, expects
            function to return True if callback was able to process
            the message, False or None otherwise.
        :param monitor_id: The id of the Monitor, will be queried
            to understand parameters of the monitor.
        """
        self.log.info("Creating Session for Monitor %s." % monitor_id)
        session = SecurePushSession(callback, monitor_id, self, self._ca_certs) \
            if self._secure else PushSession(callback, monitor_id, self)

        session.start()
        self.sessions[session.socket.fileno()] = session

        self._init_threads()
        return session
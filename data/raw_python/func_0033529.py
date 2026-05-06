def send(self, payload, token, expiration=None, priority=None, identifier=None):
        """
        Attempts to send a push message. On network failures, progagates the exception.
        It is advised to make all text in the payload dictionary unicode objects and not
        mix unicode objects and str objects. If str objects are used, they must be
        in UTF-8 encoding.
        Args:
            payload (dict): The dictionary payload of the push to send
            token (str): token to send the push to (raw, unencoded bytes)
            expiration (int, seconds): When the message becomes irrelevant (time in seconds, as from time.time())
            priority (int): Integer priority for the message as per Apple's documentation
            identifier (any): optional identifier that will be returned if the push fails.
                        This is opaque to the library and not limited to 4 bytes.
        Throws:
            BodyTooLongException: If the payload body is too long and cannot be truncated to fit
        """

        # we only use one conn at a time currently but we may as well do this...
        created_conn = False
        while not created_conn:
            if len(self.conns) == 0:
                self.conns.append(PushConnection(self, self.address, self.certfile, self.keyfile))
                created_conn = True
            conn = random.choice(self.conns)
            try:
                conn.send(payload, token, expiration=expiration, priority=priority, identifier=identifier)
                return
            except:
                logger.info("Connection died: removing")
                self.conns.remove(conn)
        raise SendFailedException()
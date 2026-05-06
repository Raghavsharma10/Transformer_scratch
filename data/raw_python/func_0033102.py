def _reallysend(self, payload, token, expiration=None, priority=None, identifier=None):
        """
        Args:
            payload (dict): The payload dictionary of the push to send
            descriptor (any): Opaque variable that is passed back to the pushbaby on failure
        """

        if not self.alive:
            raise ConnectionDeadException()
        if not self.useable:
            raise ConnectionDeadException()
        seq = self._nextSeq()
        if seq >= PushConnection.MAX_PUSHES_PER_CONNECTION:
            # IDs are 4 byte so rather than worry about wrapping IDs, just make a new connection
            # Note we don't close the connection because we want to wait to see if any errors arrive
            self._retire_connection()

        payload_str = json_for_payload(truncate(payload))
        items = ''
        items += self._apns_item(PushConnection.ITEM_DEVICE_TOKEN, token)
        items += self._apns_item(PushConnection.ITEM_PAYLOAD, payload_str)
        items += self._apns_item(PushConnection.ITEM_IDENTIFIER, seq)
        if expiration:
            items += self._apns_item(PushConnection.ITEM_EXPIRATION, expiration)
        if priority:
            items += self._apns_item(PushConnection.ITEM_PRIORITY, priority)

        apnsFrame = struct.pack("!BI", PushConnection.COMMAND_SENDPUSH, len(items)) + items

        try:
            written = 0
            while written < len(apnsFrame):
                written += self.sock.send(apnsFrame[written:])
        except:
            logger.exception("Caught exception sending push")
            raise
        self.sent[seq] = PushConnection.SentMessage(
            time.time(), token, payload, expiration, priority, identifier
        )
        self.last_push_sent = time.time()
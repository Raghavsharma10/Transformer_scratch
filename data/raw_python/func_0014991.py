def _dispatch_send(self, message):
        """
        Dispatch the different steps of sending
        """

        if self.dryrun:
            return message

        if not self.socket:
            raise GraphiteSendException(
                "Socket was not created before send"
            )

        sending_function = self._send
        if self._autoreconnect:
            sending_function = self._send_and_reconnect

        try:
            if self.asynchronous and gevent:
                gevent.spawn(sending_function, message)
            else:
                sending_function(message)
        except Exception as e:
            self._handle_send_error(e)

        return "sent {0} long message: {1}".format(len(message), message[:75])
def handle_ChannelClose(self, frame):
        """ AMQP server closed the channel with an error """
        # By docs:
        # The response to receiving a Close after sending Close must be to
        # send Close-Ok.
        #
        # No need for additional checks

        self.sender.send_CloseOK()
        exc = exceptions._get_exception_type(frame.payload.reply_code)
        self._close_all(exc)
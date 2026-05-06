def handle_ChannelCloseOK(self, frame):
        """ AMQP server closed channel as per our request """
        assert self.channel._closing, "received a not expected CloseOk"
        # Release the `close` method's future
        self.synchroniser.notify(spec.ChannelCloseOK)

        exc = ChannelClosed()
        self._close_all(exc)
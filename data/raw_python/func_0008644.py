def handle_ConnectionClose(self, frame):
        """ AMQP server closed the channel with an error """
        # Notify server we are OK to close.
        self.sender.send_CloseOK()

        exc = ConnectionClosed(frame.payload.reply_text,
                               frame.payload.reply_code)
        self._close_all(exc)
        # This will not abort transport, it will try to flush remaining data
        # asynchronously, as stated in `asyncio` docs.
        self.protocol.close()
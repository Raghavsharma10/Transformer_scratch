def write(self, value):
        """
        Queue write in kernel.
        value (bytes)
            Value to send.
        """
        aio_block = libaio.AIOBlock(
            mode=libaio.AIOBLOCK_MODE_WRITE,
            target_file=self.getEndpoint(1),
            buffer_list=[bytearray(value)],
            offset=0,
            eventfd=self.eventfd,
            onCompletion=self._onCanSend,
        )
        self._aio_send_block_list.append(aio_block)
        self._aio_context.submit([aio_block])
        if len(self._aio_send_block_list) == MAX_PENDING_WRITE_COUNT:
            self._onCannotSend()
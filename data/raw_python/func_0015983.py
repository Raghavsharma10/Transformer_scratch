def onEnable(self):
        """
        The configuration containing this function has been enabled by host.
        Endpoints become working files, so submit some read operations.
        """
        trace('onEnable')
        self._disable()
        self._aio_context.submit(self._aio_recv_block_list)
        self._real_onCanSend()
        self._enabled = True
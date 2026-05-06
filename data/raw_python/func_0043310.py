def on_open(self, callback, timeout):
        """
        Initialize a new timeout.

        :param callback: The  callback to execute when the timeout reaches the
            end of its life. May be a coroutine function.
        :param timeout: The maximum time to wait for, in seconds.
        """
        super().on_open()
        self.callback = callback
        self.timeout = timeout
        self.revive_event = asyncio.Event(loop=self.loop)
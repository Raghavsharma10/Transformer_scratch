def on_open(self, callback, period):
        """
        Initialize a new timer.

        :param callback: The function or coroutine function to call on each
            tick.
        :param period: The interval of time between two ticks.
        """
        super().on_open()
        self.callback = callback
        self.period = period
        self.reset_event = asyncio.Event(loop=self.loop)
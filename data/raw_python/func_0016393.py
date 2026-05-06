def on(self, event: str, func: Optional[Callable] = None) -> Callable:
        """
        Decorate a function to be invoked when the given event occurs.

        The function may be a coroutine.  Your function should accept **kwargs
        in case an event is triggered with unexpected kwargs.

        Example
        -------
        import asyncio
        import bottom

        client = bottom.Client(...)
        @client.on("test")
        async def func(one, two, **kwargs):
            print(one)
            print(two)
            print(kwargs)

        events.trigger("test", **{"one": 1, "two": 2, "extra": "foo"})
        loop = asyncio.get_event_loop()
        # Run all queued events
        loop.stop()
        loop.run_forever()
        """
        if func is None:
            return functools.partial(self.on, event)  # type: ignore
        wrapped = func
        if not asyncio.iscoroutinefunction(wrapped):
            wrapped = asyncio.coroutine(wrapped)
        self._event_handlers[event.upper()].append(wrapped)
        # Always return original
        return func
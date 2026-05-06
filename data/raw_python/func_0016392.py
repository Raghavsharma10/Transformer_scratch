def trigger(self, event: str, **kwargs: Any) -> None:
        """Trigger all handlers for an event to (asynchronously) execute"""
        event = event.upper()
        for func in self._event_handlers[event]:
            self.loop.create_task(func(**kwargs))
        # This will unblock anyone that is awaiting on the next loop update,
        # while still ensuring the next `await client.wait(event)` doesn't
        # immediately fire.
        async_event = self._events[event]
        async_event.set()
        async_event.clear()
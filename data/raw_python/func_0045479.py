def handle_error_event(self, act, detail, exc=SocketError):
        """
        Handle an errored event. Calls the scheduler to schedule the associated
        coroutine.
        """
        del self.tokens[act]
        self.scheduler.active.append((
            CoroutineException(exc, exc(detail)),
            act.coro
        ))
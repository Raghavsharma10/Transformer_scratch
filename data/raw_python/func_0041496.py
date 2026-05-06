def _do_dispatch(self, listeners, event_type, details):
        """Calls into listeners, handling failures and logging as needed."""
        possible_calls = len(listeners)
        call_failures = 0
        for listener in listeners:
            try:
                listener(event_type, details.copy())
            except Exception:
                self._logger.warn(
                    "Failure calling listener %s to notify about event"
                    " %s, details: %s", listener, event_type, details,
                    exc_info=True)
                call_failures += 1
        return _Notified(possible_calls,
                         possible_calls - call_failures,
                         call_failures)
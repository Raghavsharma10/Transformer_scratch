def add_periodic(self, interval: float, listener_fn):
        """Add a listener function and return listener token. Token can be closed or deleted to unlisten."""
        class PeriodicListener:
            def __init__(self, interval: float, listener_fn):
                self.interval = interval
                self.__listener_fn = listener_fn
                # the call function is very performance critical; make it fast by using a property
                # instead of a logic statement each time.
                if callable(listener_fn):
                    self.call = self.__listener_fn
                else:
                    def void(*args, **kwargs):
                        pass
                    self.call = void
                self.next_scheduled_time = time.time() + interval
            def close(self):
                self.__listener_fn = None
                def void(*args, **kwargs):
                    pass
                self.call = void
        listener = PeriodicListener(interval, listener_fn)
        def remove_listener(weak_listener):
            with self.__weak_periodic_listeners_mutex:
                self.__weak_periodic_listeners.remove(weak_listener)
        weak_listener = weakref.ref(listener, remove_listener)
        with self.__weak_periodic_listeners_mutex:
            self.__weak_periodic_listeners.append(weak_listener)
        return listener
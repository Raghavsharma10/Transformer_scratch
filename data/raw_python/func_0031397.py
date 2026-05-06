def watch(self, key, criteria, callback):
        """
        Registers a new watch under [key] (which can be used with `unwatch()`
        to remove the watch) that filters messages using [criteria] (may be a
        predicate or a 'criteria dict' [see the README for more info there]).
        Matching messages are passed to [callback], which must accept three
        arguments: the matched incoming message, this instance of
        `WatchableConnection`, and the key under which the watch was
        registered.
        """
        if hasattr(criteria, '__call__'):
            pred = criteria
        else:
            pred = lambda incoming: _match_criteria(criteria, incoming)
        with self._watches_lock:
            self._watches[key] = (pred, callback)
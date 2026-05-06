def assertSignalOrdering(self, *expected_events):
        """
        Assert that a signals were fired in a specific sequence.

        :param expected_events:
            A (varadic) list of events describing the signals that were fired
            Each element is a 3-tuple (signal, args, kwargs) that describes
            the event.

        .. note::
            If you are using :meth:`assertSignalFired()` then the return value
            of that method is a single event that can be passed to this method
        """
        expected_order = [self._events_seen.index(event)
                          for event in expected_events]
        actual_order = sorted(expected_order)
        self.assertEqual(
            expected_order, actual_order,
            "\nExpected order of fired signals:\n{}\n"
            "Actual order observed:\n{}".format(
                "\n".join(
                    "\t{}: {}".format(i, event)
                    for i, event in enumerate(expected_events, 1)),
                "\n".join(
                    "\t{}: {}".format(i, event)
                    for i, event in enumerate(
                        (self._events_seen[idx] for idx in actual_order), 1))))
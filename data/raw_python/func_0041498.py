def register(self, event_type, callback,
                 args=None, kwargs=None, details_filter=None,
                 weak=False):
        """Register a callback to be called when event of a given type occurs.

        Callback will be called with provided ``args`` and ``kwargs`` and
        when event type occurs (or on any event if ``event_type`` equals to
        :attr:`.ANY`). It will also get additional keyword argument,
        ``details``, that will hold event details provided to the
        :meth:`.notify` method (if a details filter callback is provided then
        the target callback will *only* be triggered if the details filter
        callback returns a truthy value).

        :param event_type: event type to get triggered on
        :param callback: function callback to be registered.
        :param args: non-keyworded arguments
        :type args: list
        :param kwargs: key-value pair arguments
        :type kwargs: dictionary
        :param weak: if the callback retained should be referenced via
                     a weak reference or a strong reference (defaults to
                     holding a strong reference)
        :type weak: bool

        :returns: the listener that was registered
        :rtype: :py:class:`~.Listener`
        """
        if not six.callable(callback):
            raise ValueError("Event callback must be callable")
        if details_filter is not None:
            if not six.callable(details_filter):
                raise ValueError("Details filter must be callable")
        if not self.can_be_registered(event_type):
            raise ValueError("Disallowed event type '%s' can not have a"
                             " callback registered" % event_type)
        if kwargs:
            for k in self.RESERVED_KEYS:
                if k in kwargs:
                    raise KeyError("Reserved key '%s' not allowed in "
                                   "kwargs" % k)
        with self._lock:
            if self.is_registered(event_type, callback,
                                  details_filter=details_filter):
                raise ValueError("Event callback already registered with"
                                 " equivalent details filter")
            listener = Listener(_make_ref(callback, weak=weak),
                                args=args, kwargs=kwargs,
                                details_filter=details_filter,
                                weak=weak)
            listeners = self._topics.setdefault(event_type, [])
            listeners.append(listener)
            return listener
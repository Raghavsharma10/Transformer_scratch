def disconnect(self, listener, pass_signal=False):
        """
        Disconnect an existing listener from this signal

        :param listener:
            The listener (callable) to remove
        :param pass_signal:
            An optional argument that controls if the signal object is
            explicitly passed to this listener when it is being fired.
            If enabled, a ``signal=`` keyword argument is passed to the
            listener function.

            Here, this argument simply aids in disconnecting the right
            listener. Make sure to pass the same value as was passed to
            :meth:`connect()`
        :raises ValueError:
            If the listener (with the same value of pass_signal) is not present
        :returns:
            None
        """
        info = listenerinfo(listener, pass_signal)
        self._listeners.remove(info)
        _logger.debug(
            "disconnect %r from %r", str(listener), self._name)
        if inspect.ismethod(listener):
            listener_object = listener.__self__
            if hasattr(listener_object, "__listeners__"):
                listener_object.__listeners__[listener].remove(self)
                # Remove the listener from the list if any signals connected
                if (len(listener_object.__listeners__[listener])) == 0:
                    del listener_object.__listeners__[listener]
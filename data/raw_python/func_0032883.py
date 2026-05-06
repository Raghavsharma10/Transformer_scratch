def wait_actions(self, actions, wait_interval=None, wait_time=None):
        r"""
        Poll the server periodically until all actions in ``actions`` have
        either completed or errored out, yielding each `Action`'s final value
        as it ends.

        If ``wait_time`` is exceeded, a `WaitTimeoutError` (containing any
        remaining in-progress actions) is raised.

        If a `KeyboardInterrupt` is caught, any remaining actions are returned
        immediately without waiting for completion.

        .. versionchanged:: 0.2.0
            Raises `WaitTimeoutError` on timeout

        :param iterable actions: an iterable of `Action`\ s and/or other values
            that are acceptable arguments to :meth:`fetch_action`
        :param number wait_interval: how many seconds to sleep between
            requests; defaults to :attr:`wait_interval` if not specified or
            `None`
        :param number wait_time: the total number of seconds after which the
            method will raise an error if any actions have not yet completed,
            or a negative number to wait indefinitely; defaults to
            :attr:`wait_time` if not specified or `None`
        :rtype: generator of `Action`\ s
        :raises DOAPIError: if the API endpoint replies with an error
        :raises WaitTimeoutError: if ``wait_time`` is exceeded
        """
        return self._wait(map(self._action, actions), "done", True,
                          wait_interval, wait_time)
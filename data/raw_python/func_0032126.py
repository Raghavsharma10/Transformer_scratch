def wait(self, wait_interval=None, wait_time=None):
        """
        Poll the server periodically until the action has either completed or
        errored out and return its final state.

        If ``wait_time`` is exceeded, a `WaitTimeoutError` (containing the
        action's most recently fetched state) is raised.

        If a `KeyboardInterrupt` is caught, the action's most recently fetched
        state is returned immediately without waiting for completion.

        .. versionchanged:: 0.2.0
            Raises `WaitTimeoutError` on timeout

        :param number wait_interval: how many seconds to sleep between
            requests; defaults to the `doapi` object's
            :attr:`~doapi.wait_interval` if not specified or `None`
        :param number wait_time: the total number of seconds after which the
            method will raise an error if the action has not yet completed, or
            a negative number to wait indefinitely; defaults to the `doapi`
            object's :attr:`~doapi.wait_time` if not specified or `None`
        :return: the action's final state
        :rtype: Action
        :raises DOAPIError: if the API endpoint replies with an error
        :raises WaitTimeoutError: if ``wait_time`` is exceeded
        """
        return next(self.doapi_manager.wait_actions([self], wait_interval,
                                                            wait_time))
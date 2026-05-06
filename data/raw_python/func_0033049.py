def wait(self, status=None, locked=None, wait_interval=None,
             wait_time=None):
        """
        Poll the server periodically until the droplet has reached some final
        state.  If ``status`` is non-`None`, ``wait`` will wait for the
        droplet's ``status`` field to equal the given value.  If ``locked`` is
        non-`None`, `wait` will wait for the droplet's ``locked`` field to
        equal (the truth value of) the given value.  Exactly one of ``status``
        and ``locked`` must be non-`None`.

        If ``wait_time`` is exceeded, a `WaitTimeoutError` (containing the
        droplet's most recently fetched state) is raised.

        If a `KeyboardInterrupt` is caught, the droplet's most recently fetched
        state is returned immediately without waiting for completion.

        .. versionchanged:: 0.2.0
            Raises `WaitTimeoutError` on timeout

        .. versionchanged:: 0.2.0
            ``locked`` parameter added

        .. versionchanged:: 0.2.0
            No longer waits for latest action to complete

        :param status: When non-`None`, the desired value for the ``status``
            field of the droplet, which should be one of
            `Droplet.STATUS_ACTIVE`, `Droplet.STATUS_ARCHIVE`,
            `Droplet.STATUS_NEW`, and `Droplet.STATUS_OFF`.  (For the sake of
            forwards-compatibility, any other value is accepted as well.)
        :type status: string or `None`
        :param locked: When non-`None`, the desired value for the ``locked``
            field of the droplet
        :type locked: `bool` or `None`
        :param number wait_interval: how many seconds to sleep between
            requests; defaults to the `doapi` object's
            :attr:`~doapi.wait_interval` if not specified or `None`
        :param number wait_time: the total number of seconds after which the
            method will raise an error if the droplet has not yet completed, or
            a negative number to wait indefinitely; defaults to the `doapi`
            object's :attr:`~doapi.wait_time` if not specified or `None`
        :return: the droplet's final state
        :rtype: Droplet
        :raises TypeError: if both or neither of ``status`` & ``locked`` are
            defined
        :raises DOAPIError: if the API endpoint replies with an error
        :raises WaitTimeoutError: if ``wait_time`` is exceeded
        """
        return next(self.doapi_manager.wait_droplets([self], status, locked,
                                                     wait_interval, wait_time))
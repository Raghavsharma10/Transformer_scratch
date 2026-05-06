def wait_droplets(self, droplets, status=None, locked=None,
                      wait_interval=None, wait_time=None):
        r"""
        Poll the server periodically until all droplets in ``droplets`` have
        reached some final state, yielding each `Droplet`'s final value when
        it's done.  If ``status`` is non-`None`, ``wait_droplets`` will wait
        for each droplet's ``status`` field to equal the given value.  If
        ``locked`` is non-`None`, ``wait_droplets`` will wait for each
        droplet's ``locked`` field to equal (the truth value of) the given
        value.  Exactly one of ``status`` and ``locked`` must be non-`None`.

        If ``wait_time`` is exceeded, a `WaitTimeoutError` (containing any
        remaining in-progress droplets) is raised.

        If a `KeyboardInterrupt` is caught, any remaining droplets are returned
        immediately without waiting for completion.

        .. versionchanged:: 0.2.0
            Raises `WaitTimeoutError` on timeout

        .. versionchanged:: 0.2.0
            ``locked`` parameter added

        .. versionchanged:: 0.2.0
            No longer waits for actions to complete

        :param iterable droplets: an iterable of `Droplet`\ s and/or other
            values that are acceptable arguments to :meth:`fetch_droplet`
        :param status: When non-`None`, the desired value for the ``status``
            field of each `Droplet`, which should be one of
            `Droplet.STATUS_ACTIVE`, `Droplet.STATUS_ARCHIVE`,
            `Droplet.STATUS_NEW`, and `Droplet.STATUS_OFF`.  (For the sake of
            forwards-compatibility, any other value is accepted as well.)
        :type status: string or `None`
        :param locked: When non-`None`, the desired value for the ``locked``
            field of each `Droplet`
        :type locked: `bool` or `None`
        :param number wait_interval: how many seconds to sleep between
            requests; defaults to :attr:`wait_interval` if not specified or
            `None`
        :param number wait_time: the total number of seconds after which the
            method will raise an error if any droplets have not yet completed,
            or a negative number to wait indefinitely; defaults to
            :attr:`wait_time` if not specified or `None`
        :rtype: generator of `Droplet`\ s
        :raises TypeError: if both or neither of ``status`` & ``locked`` are
            defined
        :raises DOAPIError: if the API endpoint replies with an error
        :raises WaitTimeoutError: if ``wait_time`` is exceeded
        """
        if (status is None) == (locked is None):
            ### TODO: Is TypeError the right type of error?
            raise TypeError('Exactly one of "status" and "locked" must be'
                            ' specified')
        droplets = map(self._droplet, droplets)
        if status is not None:
            return self._wait(droplets, "status", status, wait_interval,
                              wait_time)
        if locked is not None:
            return self._wait(droplets, "locked", bool(locked), wait_interval,
                              wait_time)
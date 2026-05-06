def _wait(self, objects, attr, value, wait_interval=None, wait_time=None):
        r"""
        Calls the ``fetch`` method of each object in ``objects`` periodically
        until the ``attr`` attribute of each one equals ``value``, yielding the
        final state of each object as soon as it satisfies the condition.

        If ``wait_time`` is exceeded, a `WaitTimeoutError` (containing any
        remaining in-progress objects) is raised.

        If a `KeyboardInterrupt` is caught, any remaining objects are returned
        immediately without waiting for completion.

        .. versionchanged:: 0.2.0
            Raises `WaitTimeoutError` on timeout

        :param iterable objects: an iterable of `Resource`\ s with ``fetch``
            methods
        :param string attr: the attribute to watch
        :param value: the value of ``attr`` to wait for
        :param number wait_interval: how many seconds to sleep between
            requests; defaults to :attr:`wait_interval` if not specified or
            `None`
        :param number wait_time: the total number of seconds after which the
            method will raise an error if any objects have not yet completed,
            or a negative number to wait indefinitely; defaults to
            :attr:`wait_time` if not specified or `None`
        :rtype: generator
        :raises DOAPIError: if the API endpoint replies with an error
        :raises WaitTimeoutError: if ``wait_time`` is exceeded
        """

        objects = list(objects)
        if not objects:
            return
        if wait_interval is None:
            wait_interval = self.wait_interval
        if wait_time < 0:
            end_time = None
        else:
            if wait_time is None:
                wait_time = self.wait_time
            if wait_time is None or wait_time < 0:
                end_time = None
            else:
                end_time = time() + wait_time
        while end_time is None or time() < end_time:
            loop_start = time()
            next_objs = []
            for o in objects:
                obj = o.fetch()
                if getattr(obj, attr, None) == value:
                    yield obj
                else:
                    next_objs.append(obj)
            objects = next_objs
            if not objects:
                break
            loop_end = time()
            time_left = wait_interval - (loop_end - loop_start)
            if end_time is not None:
                time_left = min(time_left, end_time - loop_end)
            if time_left > 0:
                try:
                    sleep(time_left)
                except KeyboardInterrupt:
                    for o in objects:
                        yield o
                    return
        if objects:
            raise WaitTimeoutError(objects, attr, value, wait_interval,
                                   wait_time)
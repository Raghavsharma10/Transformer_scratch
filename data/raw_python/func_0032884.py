def wait_actions_on_objects(self, objects, wait_interval=None,
                                               wait_time=None):
        """
        .. versionadded:: 0.2.0

        Poll the server periodically until the most recent action on each
        resource in ``objects`` has finished, yielding each resource's final
        state when the corresponding action is done.

        If ``wait_time`` is exceeded, a `WaitTimeoutError` (containing any
        remaining in-progress actions) is raised.

        If a `KeyboardInterrupt` is caught, any remaining actions are returned
        immediately without waiting for completion.

        :param iterable objects: an iterable of resource objects that have
            ``fetch_last_action`` methods
        :param number wait_interval: how many seconds to sleep between
            requests; defaults to :attr:`wait_interval` if not specified or
            `None`
        :param number wait_time: the total number of seconds after which the
            method will raise an error if any actions have not yet completed,
            or a negative number to wait indefinitely; defaults to
            :attr:`wait_time` if not specified or `None`
        :rtype: generator of objects
        :raises DOAPIError: if the API endpoint replies with an error
        :raises WaitTimeoutError: if ``wait_time`` is exceeded
        """
        acts = []
        for o in objects:
            a = o.fetch_last_action()
            if a is None:
                yield o
            else:
                acts.append(a)
        for a in self.wait_actions(acts, wait_interval, wait_time):
            yield a.fetch_resource()
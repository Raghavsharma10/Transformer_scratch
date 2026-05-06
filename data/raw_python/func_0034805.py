def _get_tracker(self, resource):
        """
        Return the resource tracker that is tracking ``resource``.

        :param resource: A resource.
        :return: A resource tracker.
        :rtype: :class:`_ResourceTracker`
        """
        with self._lock:
            for rt in self._reference_queue:
                if rt is not None and resource is rt.resource:
                    return rt

        raise UnknownResourceError('Resource not created by pool')
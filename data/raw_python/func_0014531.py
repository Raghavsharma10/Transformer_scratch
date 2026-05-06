def clone_pod(self, source, dest, **kwargs):
        """Clone an existing pod to a new one.

        :param source: Name of the pod the be cloned.
        :type source: str
        :param dest: Name of the target pod to clone into
        :type dest: str
        :param \*\*kwargs: See the REST API Guide on your array for the
                           documentation on the request:
                           **POST pod/:pod**
        :type \*\*kwargs: optional

        :returns: A dictionary describing the created pod
        :rtype: ResponseDict

        .. note::

            Requires use of REST API 1.13 or later.
        """
        data = {"source": source}
        data.update(kwargs)
        return self._request("POST", "pod/{0}".format(dest), data)
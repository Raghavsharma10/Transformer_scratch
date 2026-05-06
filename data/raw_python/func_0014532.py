def remove_pod(self, pod, array, **kwargs):
        """Remove arrays from a pod.

        :param pod: Name of the pod.
        :type pod: str
        :param array: Array to remove from pod.
        :type array: str
        :param \*\*kwargs: See the REST API Guide on your array for the
                           documentation on the request:
                           **DELETE pod/:pod**/array/:array**
        :type \*\*kwargs: optional
        :returns: A dictionary mapping "name" to pod and "array" to the pod's
                  new array list.
        :rtype: ResponseDict

        .. note::

            Requires use of REST API 1.13 or later.
        """
        return self._request("DELETE", "pod/{0}/array/{1}".format(pod, array), kwargs)
def eradicate_pgroup(self, pgroup, **kwargs):
        """Eradicate a destroyed pgroup.

        :param pgroup: Name of pgroup to be eradicated.
        :type pgroup: str
        :param \*\*kwargs: See the REST API Guide on your array for the
                           documentation on the request:
                           **DELETE pgroup/:pgroup**
        :type \*\*kwargs: optional

        :returns: A dictionary mapping "name" to pgroup.
        :rtype: ResponseDict

        .. note::

            Requires use of REST API 1.2 or later.

        """
        eradicate = {"eradicate": True}
        eradicate.update(kwargs)
        return self._request("DELETE", "pgroup/{0}".format(pgroup), eradicate)
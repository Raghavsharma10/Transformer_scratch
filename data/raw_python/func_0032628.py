def _record(self, obj):
        """
        Construct a `DomainRecord` object belonging to the domain's `doapi`
        object.  ``obj`` may be a domain record ID, a dictionary of domain
        record fields, or another `DomainRecord` object (which will be
        shallow-copied).  The resulting `DomainRecord` will only contain the
        information in ``obj``; no data will be sent to or from the API
        endpoint.

        :type obj: integer, `dict`, or `DomainRecord`
        :rtype: DomainRecord
        """
        return DomainRecord(obj, domain=self, doapi_manager=self.doapi_manager)
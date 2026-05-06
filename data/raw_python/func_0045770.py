def get_log_entry(self, log_entry_id):
        """Gets the ``LogEntry`` specified by its ``Id``.

        In plenary mode, the exact ``Id`` is found or a ``NotFound``
        results. Otherwise, the returned ``LogEntry`` may have a
        different ``Id`` than requested, such as the case where a
        duplicate ``Id`` was assigned to a ``LogEntry`` and retained for
        compatibility.

        arg:    log_entry_id (osid.id.Id): the ``Id`` of the
                ``LogEntry`` to retrieve
        return: (osid.logging.LogEntry) - the returned ``LogEntry``
        raise:  NotFound - no ``LogEntry`` found with the given ``Id``
        raise:  NullArgument - ``log_entry_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resource
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('logging',
                                         collection='LogEntry',
                                         runtime=self._runtime)
        result = collection.find_one(
            dict({'_id': ObjectId(self._get_id(log_entry_id, 'logging').get_identifier())},
                 **self._view_filter()))
        return objects.LogEntry(osid_object_map=result, runtime=self._runtime, proxy=self._proxy)
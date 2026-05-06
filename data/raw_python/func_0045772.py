def get_log_entries_by_genus_type(self, log_entry_genus_type):
        """Gets a ``LogEntryList`` corresponding to the given log entry genus ``Type`` which doe snot include entries of genus types derived form the specified ``Type``.

        In plenary mode, the returned list contains all known entries or
        an error results. Otherwise, the returned list may contain only
        those entries that are accessible through this session

        arg:    log_entry_genus_type (osid.type.Type): a log entry genus
                type
        return: (osid.logging.LogEntryList) - the returned ``LogEntry``
                list
        raise:  NullArgument - ``log_entry_genus_type`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_genus_type
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('logging',
                                         collection='LogEntry',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'genusTypeId': str(log_entry_genus_type)},
                 **self._view_filter())).sort('_id', DESCENDING)
        return objects.LogEntryList(result, runtime=self._runtime, proxy=self._proxy)
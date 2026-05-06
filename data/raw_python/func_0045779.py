def delete_log_entry(self, log_entry_id):
        """Deletes a ``LogEntry``.

        arg:    log_entry_id (osid.id.Id): the ``Id`` of the
                ``log_entry_id`` to remove
        raise:  NotFound - ``log_entry_id`` not found
        raise:  NullArgument - ``log_entry_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.delete_resource_template
        collection = JSONClientValidated('logging',
                                         collection='LogEntry',
                                         runtime=self._runtime)
        if not isinstance(log_entry_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        log_entry_map = collection.find_one(
            dict({'_id': ObjectId(log_entry_id.get_identifier())},
                 **self._view_filter()))

        objects.LogEntry(osid_object_map=log_entry_map, runtime=self._runtime, proxy=self._proxy)._delete()
        collection.delete_one({'_id': ObjectId(log_entry_id.get_identifier())})
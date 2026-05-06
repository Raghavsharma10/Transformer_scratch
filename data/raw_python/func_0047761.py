def register_for_deleted_resource(self, resource_id):
        """Registers for notification of a deleted resource.

        ``ResourceReceiver.deletedResources()`` is invoked when the
        specified resource is deleted or removed from this bin.

        arg:    resource_id (osid.id.Id): the ``Id`` of the ``Resource``
                to monitor
        raise:  NullArgument - ``resource_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceNotificationSession.register_for_deleted_resource
        if not MONGO_LISTENER.receivers[self._ns][self._receiver]['d']:
            MONGO_LISTENER.receivers[self._ns][self._receiver]['d'] = []
        if isinstance(MONGO_LISTENER.receivers[self._ns][self._receiver]['d'], list):
            MONGO_LISTENER.receivers[self._ns][self._receiver]['d'].append(resource_id.get_identifier())
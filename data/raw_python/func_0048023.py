def register_for_deleted_item(self, item_id):
        """Registers for notification of a deleted assessment item.

        ``ItemReceiver.deletedItems()`` is invoked when the specified
        assessment item is removed from the assessment bank.

        arg:    item_id (osid.id.Id): the ``Id`` of the ``Item`` to
                monitor
        raise:  NotFound - an ``Item`` was not found identified by the
                given ``Id``
        raise:  NullArgument - ``item_id is null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceNotificationSession.register_for_deleted_resource
        if not MONGO_LISTENER.receivers[self._ns][self._receiver]['d']:
            MONGO_LISTENER.receivers[self._ns][self._receiver]['d'] = []
        if isinstance(MONGO_LISTENER.receivers[self._ns][self._receiver]['d'], list):
            MONGO_LISTENER.receivers[self._ns][self._receiver]['d'].append(item_id.get_identifier())
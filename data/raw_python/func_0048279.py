def get_resource_notification_session_for_bin(self, resource_receiver, bin_id, proxy):
        """Gets the resource notification session for the given bin.

        arg:    resource_receiver (osid.resource.ResourceReceiver):
                notification callback
        arg:    bin_id (osid.id.Id): the ``Id`` of the bin
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.resource.ResourceNotificationSession) - ``a
                ResourceNotificationSession``
        raise:  NotFound - ``bin_id`` not found
        raise:  NullArgument - ``resource_receiver, bin_id`` or
                ``proxy`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_resource_notification()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_resource_notfication()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_resource_notification():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.ResourceNotificationSession(catalog_id=bin_id, proxy=proxy, runtime=self._runtime, receiver=resource_receiver)
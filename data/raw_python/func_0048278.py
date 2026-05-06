def get_resource_notification_session(self, resource_receiver, proxy):
        """Gets the resource notification session for the given bin.

        arg:    resource_receiver (osid.resource.ResourceReceiver):
                notification callback
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.resource.ResourceNotificationSession) - ``a
                ResourceNotificationSession``
        raise:  NullArgument - ``resource_receiver`` or ``proxy`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_resource_notification()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_resource_notification()`` is ``true``.*

        """
        if not self.supports_resource_notification():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ResourceNotificationSession(proxy=proxy, runtime=self._runtime, receiver=resource_receiver)
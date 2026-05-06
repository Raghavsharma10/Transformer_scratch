def get_item_notification_session(self, item_receiver):
        """Gets the notification session for notifications pertaining to item changes.

        arg:    item_receiver (osid.assessment.ItemReceiver): the item
                receiver interface
        return: (osid.assessment.ItemNotificationSession) - an
                ``ItemNotificationSession``
        raise:  NullArgument - ``item_receiver`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_item_notification()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_item_notification()`` is ``true``.*

        """
        if not self.supports_item_notification():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ItemNotificationSession(runtime=self._runtime, receiver=item_receiver)
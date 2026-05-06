def get_item_notification_session_for_bank(self, item_receiver, bank_id):
        """Gets the ``OsidSession`` associated with the item notification service for the given bank.

        arg:    item_receiver (osid.assessment.ItemReceiver): the item
                receiver interface
        arg:    bank_id (osid.id.Id): the ``Id`` of the bank
        return: (osid.assessment.AssessmentNotificationSession) - ``an
                _item_notification_session``
        raise:  NotFound - ``bank_id`` not found
        raise:  NullArgument - ``item_receiver`` or ``bank_id`` is
                ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_item_notification()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_item_notification()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_item_notification():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.ItemNotificationSession(bank_id, runtime=self._runtime, receiver=item_receiver)
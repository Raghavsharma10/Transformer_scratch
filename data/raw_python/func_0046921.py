def get_asset_notification_session(self, asset_receiver):
        """Gets the notification session for notifications pertaining to asset changes.

        arg:    asset_receiver (osid.repository.AssetReceiver): the
                notification callback
        return: (osid.repository.AssetNotificationSession) - an
                ``AssetNotificationSession``
        raise:  NullArgument - ``asset_receiver`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_asset_notification()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_asset_notification()`` is ``true``.*

        """
        if not self.supports_asset_notification():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssetNotificationSession(runtime=self._runtime, receiver=asset_receiver)
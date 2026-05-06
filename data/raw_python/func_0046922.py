def get_asset_notification_session_for_repository(self, asset_receiver, repository_id):
        """Gets the asset notification session for the given repository.

        arg:    asset_receiver (osid.repository.AssetReceiver): the
                notification callback
        arg:    repository_id (osid.id.Id): the ``Id`` of the repository
        return: (osid.repository.AssetNotificationSession) - an
                ``AssetNotificationSession``
        raise:  NotFound - ``repository_id`` not found
        raise:  NullArgument - ``asset_receiver`` or ``repository_id``
                is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_asset_notification()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_asset_notfication()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_asset_notification():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.AssetNotificationSession(repository_id, runtime=self._runtime, receiver=asset_receiver)
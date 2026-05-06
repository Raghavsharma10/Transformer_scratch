def delete_asset(self, asset_id=None):
        """Deletes an ``Asset``.

        arg:    asset_id (osid.id.Id): the ``Id`` of the ``Asset`` to
                remove
        raise:  NotFound - ``asset_id`` not found
        raise:  NullArgument - ``asset_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from awsosid template for -
        # osid.resource.ResourceAdminSession.delete_resource_template
        # clean up AWS
        asset = self._asset_lookup_session.get_asset(asset_id)
        for ac in asset.asset_contents:
            self.delete_asset_content(ac.ident)
        self._provider_session.delete_asset(asset_id)
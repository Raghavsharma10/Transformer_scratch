def grant_scim_permissions(self, client_id, read=False, write=False,
            create=False, userids=False, zones=False, invite=False,
            openid=False):
        """
        Grant the given client_id permissions for managing users.  System
        for Cross-domain Identity Management (SCIM) are required for accessing
        /Users and /Groups endpoints of UAA.

        - scim.read: scope for read access to all SCIM endpoints
        - scim.write: scope for write access to all SCIM endpoints
        - scim.create: scope to create/invite users and verify an account only
        - scim.userids: scope for id and username+origin conversion
        - scim.zones: scope for group management of users only
        - scim.invite: scope to participate in invitations
        - openid: scope to access /userinfo

        """
        self.assert_has_permission('clients.admin')

        perms = []
        if read:
            perms.append('scim.read')

        if write:
            perms.append('scim.write')

        if create:
            perms.append('scim.create')

        if userids:
            perms.append('scim.userids')

        if zones:
            perms.append('scim.zones')

        if invite:
            perms.append('scim.invite')

        if openid:
            perms.append('openid')

        if perms:
            self.update_client_grants(client_id, scope=perms, authorities=perms)
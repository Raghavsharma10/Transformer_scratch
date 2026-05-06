def grant_client_permissions(self, client_id, admin=False, write=False,
            read=False, secret=False):
        """
        Grant the given client_id permissions for managing clients.

        - clients.admin: super user scope to create, modify, delete
        - clients.write: scope ot create and modify clients
        - clients.read: scope to read info about clients
        - clients.secret: scope to change password of a client

        """
        self.assert_has_permission('clients.admin')

        perms = []
        if admin:
            perms.append('clients.admin')

        if write or admin:
            perms.append('clients.write')

        if read or admin:
            perms.append('clients.read')

        if secret or admin:
            perms.append('clients.secret')

        if perms:
            self.update_client_grants(client_id, scope=perms,
                    authorities=perms)
def create_space(self, space_name, add_users=True):
        """
        Create a new space with the given name in the current target
        organization.
        """
        body = {
            'name': space_name,
            'organization_guid': self.api.config.get_organization_guid()
        }

        # MAINT: may need to do this more generally later
        if add_users:
            space_users = []
            org_users = self.org.get_users()
            for org_user in org_users['resources']:
                guid = org_user['metadata']['guid']
                space_users.append(guid)

            body['manager_guids'] = space_users
            body['developer_guids'] = space_users

        return self.api.post('/v2/spaces', body)
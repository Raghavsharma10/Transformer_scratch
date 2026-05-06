def users(self):
        """ returns all users that have permissions for this resource"""
        return sa.orm.relationship(
            "User",
            secondary="users_resources_permissions",
            passive_deletes=True,
            passive_updates=True,
        )
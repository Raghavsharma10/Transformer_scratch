def groups_dynamic(self):
        """ returns dynamic relationship for groups - allowing for
        filtering of data """
        return sa.orm.relationship(
            "Group",
            secondary="users_groups",
            lazy="dynamic",
            passive_deletes=True,
            passive_updates=True,
        )
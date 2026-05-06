def resources_dynamic(self):
        """ Returns all resources directly owned by group, can be used to assign
        ownership of new resources::

            user.resources.append(resource) """
        return sa.orm.relationship(
            "Resource",
            cascade="all",
            passive_deletes=True,
            passive_updates=True,
            lazy="dynamic",
        )
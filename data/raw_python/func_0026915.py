def groups_with_resources(cls, instance):
        """
        Returns a list of groups users belongs to with eager loaded
        resources owned by those groups

        :param instance:
        :return:
        """
        return instance.groups_dynamic.options(
            sa.orm.eagerload(cls.models_proxy.Group.resources)
        )
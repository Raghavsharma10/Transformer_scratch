def get_resource_groups(self):
        """
        :return: dictionary {resource group id: object} of all resource groups.
        """

        resource_groups = {r.index: r for r in self.get_objects_by_type('resourceGroupEx')}
        return OrderedDict(sorted(resource_groups.items()))
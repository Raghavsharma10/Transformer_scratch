def add_from_child(self, resource, **kwargs):
        """ Add a resource with its all children resources to the current
        resource.
        """

        new_resource = self.add(
            resource.member_name, resource.collection_name, **kwargs)
        for child in resource.children:
            new_resource.add_from_child(child, **kwargs)
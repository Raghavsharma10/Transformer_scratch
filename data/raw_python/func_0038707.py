def save(self, request, resource=None, **kwargs):
        """Create a resource."""
        resources = resource if isinstance(resource, list) else [resource]
        for obj in resources:
            obj.save()
        return resource
async def delete(self, request, resource=None, **kwargs):
        """Delete a resource."""
        if resource is None:
            raise RESTNotFound(reason='Resource not found')
        self.collection.remove(resource)
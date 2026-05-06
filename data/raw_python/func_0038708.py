async def delete(self, request, resource=None, **kwargs):
        """Delete a resource.

        Supports batch delete.
        """
        if resource:
            resources = [resource]
        else:
            data = await self.parse(request)
            if data:
                resources = list(self.collection.where(self.meta.model_pk << data))

        if not resources:
            raise RESTNotFound(reason='Resource not found')

        for resource in resources:
            resource.delete_instance()
async def put(self, request, resource=None, **kwargs):
        """Update a resource.

        ---
        parameters:
            - name: resource
              in: path
              type: string
        """
        if resource is None:
            raise RESTNotFound(reason='Resource not found')

        return await self.post(request, resource=resource, **kwargs)
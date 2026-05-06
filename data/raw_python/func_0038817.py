async def get(self, request, resource=None, **kwargs):
        """Get resource or collection of resources.

        ---
        parameters:
            - name: resource
              in: path
              type: string

        """
        if resource is not None and resource != '':
            return self.to_simple(request, resource, **kwargs)

        return self.to_simple(request, self.collection, many=True, **kwargs)
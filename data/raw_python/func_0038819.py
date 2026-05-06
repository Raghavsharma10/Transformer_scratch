async def post(self, request, resource=None, **kwargs):
        """Create a resource."""
        resource = await self.load(request, resource=resource, **kwargs)
        resource = await self.save(request, resource=resource, **kwargs)
        return self.to_simple(request, resource, many=isinstance(resource, list), **kwargs)
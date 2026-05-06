async def load(self, request, resource=None, **kwargs):
        """Load resource from given data."""
        schema = self.get_schema(request, resource=resource, **kwargs)
        data = await self.parse(request)
        resource, errors = schema.load(
            data, partial=resource is not None, many=isinstance(data, list))
        if errors:
            raise RESTBadRequest(reason='Bad request', json={'errors': errors})
        return resource
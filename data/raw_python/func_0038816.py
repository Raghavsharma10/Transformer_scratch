async def filter(self, request, **kwargs):
        """Filter collection."""
        try:
            data = loads(request.query.get(VAR_WHERE))
        except (ValueError, TypeError):
            return self.collection

        self.filters, collection = self.meta.filters.filter(
            data, self.collection, resource=self, **kwargs)

        return collection
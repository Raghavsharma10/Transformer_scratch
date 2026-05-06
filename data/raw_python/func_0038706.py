def paginate(self, request, offset=0, limit=None):
        """Paginate queryset."""
        return self.collection.offset(offset).limit(limit), self.collection.count()
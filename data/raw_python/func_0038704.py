def get_one(self, request, **kwargs):
        """Load a resource."""
        resource = request.match_info.get(self.name)
        if not resource:
            return None

        try:
            return self.collection.where(self.meta.model_pk == resource).get()
        except Exception:
            raise RESTNotFound(reason='Resource not found.')
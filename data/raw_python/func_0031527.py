def fetch_object(self, model_id):
        """Fetch the model by its ID."""
        pk_field_instance = getattr(self.object_class, self.pk_field)
        qs = self.object_class.query.filter(pk_field_instance == model_id)
        model = qs.one_or_none()
        if not model:
            raise ReferenceNotFoundError
        return model
def get_source_fields(self, instance):
        """
        Returns a list of the source fields for the given instance.
        """
        return set(src.image_field
                   for src in self._source_groups
                   if isinstance(instance, src.model_class))
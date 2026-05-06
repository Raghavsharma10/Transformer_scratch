def filter(self, *args, **kwargs):
        """filter lets django managers use `objects.filter` on a hashable object."""
        obj = kwargs.pop(self.object_property_name, None)
        if obj is not None:
            kwargs['object_hash'] = self.model._compute_hash(obj)
        return super().filter(*args, **kwargs)
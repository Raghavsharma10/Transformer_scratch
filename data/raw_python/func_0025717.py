def _extract_model_params(self, defaults, **kwargs):
        """this method allows django managers use `objects.get_or_create` and
        `objects.update_or_create` on a hashable object.
        """
        obj = kwargs.pop(self.object_property_name, None)
        if obj is not None:
            kwargs['object_hash'] = self.model._compute_hash(obj)
        lookup, params = super()._extract_model_params(defaults, **kwargs)
        if obj is not None:
            params[self.object_property_name] = obj
            del params['object_hash']
        return lookup, params
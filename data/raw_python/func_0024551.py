def get_object_keys(self, wfi_role=None):
        """returns object keys according to task definition
        which can be explicitly selected one object (self.object_key) or
        result of a queryset filter.

        Returns:
            list of object keys
        """
        if self.object_key:
            return [self.object_key]
        if self.object_query_code:
            model = model_registry.get_model(self.object_type)
            return [m.key for m in
                    self.get_model_objects(model, wfi_role, **self.get_object_query_dict())]
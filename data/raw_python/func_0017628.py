def field_dict(self, model):
        """
        Helper function that returns a dictionary of all fields in the given
        model. If self.field_filter is set, it only includes the fields that
        match the filter.
        """
        if self.field_filter:
            return dict(
                [(f.name, f) for f in model._meta.fields
                 if self.field_filter(f)]
            )
        else:
            return dict(
                [(f.name, f) for f in model._meta.fields
                 if not f.rel and
                    not f.primary_key and
                    not f.unique and
                    not isinstance(f, (models.AutoField, models.TextField))]
            )
def field_dict(self, model):
        """
        Helper function that returns a dictionary of all DateFields or
        DateTimeFields in the given model. If self.field_names is set,
        it takes that into account when building the dictionary.
        """
        if self.field_names is None:
            return dict([(f.name, f) for f in model._meta.fields
                         if isinstance(f, models.DateField)])
        else:
            return dict([(f.name, f)
                         for f in model._meta.fields
                         if isinstance(f, models.DateField) and
                            (f.name in self.field_names)])
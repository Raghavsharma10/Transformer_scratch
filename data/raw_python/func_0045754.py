def clean(self, value):
        """
        Clean the field values.
        """
        if not self.create:
            # No new value can be created, use the regular clean field
            return super(AgnocompleteModelMultipleField, self).clean(value)

        # We have to do this here before the call to "super".
        # It'll be called again, but we can't find a way to "pre_clean" the
        # field value before pushing it into the parent class "clean()" method.
        value = self.clear_list_value(value)
        # Split the actual values with the potential new values
        # Numeric values will always be considered as PKs
        pks = [v for v in value if v.isdigit()]
        self._new_values = [v for v in value if not v.isdigit()]

        qs = super(AgnocompleteModelMultipleField, self).clean(pks)

        return qs
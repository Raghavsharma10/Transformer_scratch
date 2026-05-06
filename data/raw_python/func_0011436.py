def mutate(self, field):
        """Mutate the given field, modifying it directly. This is not
        intended to preserve the value of the field.

        :field: The pfp.fields.Field instance that will receive the new value
        """
        new_val = self.next_val(field)
        field._pfp__set_value(new_val)
        return field
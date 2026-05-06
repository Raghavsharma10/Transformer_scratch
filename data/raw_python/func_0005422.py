def update(self, dict):
        """Set all field values from a dictionary.

        For any key in `dict` that is also a field to store tags the
        method retrieves the corresponding value from `dict` and updates
        the `MediaFile`. If a key has the value `None`, the
        corresponding property is deleted from the `MediaFile`.
        """
        for field in self.sorted_fields():
            if field in dict:
                if dict[field] is None:
                    delattr(self, field)
                else:
                    setattr(self, field, dict[field])
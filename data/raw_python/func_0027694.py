def save(self, commit=True):
        """save the instance or create a new one.."""

        # walk through the document fields
        for field_name, field in iter_valid_fields(self._meta):
            setattr(self.instance, field_name, self.cleaned_data.get(field_name))

        if commit:
            self.instance.save()

        return self.instance
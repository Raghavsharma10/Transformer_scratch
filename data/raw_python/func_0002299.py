def value_from_object(self, obj):
        """
        Internal Django method, used to return the placeholder ID when exporting the model instance.
        """
        try:
            # not using self.attname, access the descriptor instead.
            placeholder = getattr(obj, self.name)
        except Placeholder.DoesNotExist:
            return None   # Still allow ModelForm / admin to open and create a new Placeholder if the table was truncated.

        return placeholder.id if placeholder else None
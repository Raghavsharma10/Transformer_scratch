def lookup_field_orderable(self, field):
        """
        Returns whether the passed in field is sortable or not, by default all 'raw' fields, that
        is fields that are part of the model are sortable.
        """
        try:
            self.model._meta.get_field_by_name(field)
            return True
        except Exception:
            # that field doesn't exist, so not sortable
            return False
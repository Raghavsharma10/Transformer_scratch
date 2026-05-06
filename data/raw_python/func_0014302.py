def _get_record(self):
        """Fetch a given record.

        Handles fetching a record from the database along with throwing an
        appropriate instance of ``AJAXError`.
        """
        if not self.pk:
            raise AJAXError(400, _('Invalid request for record.'))

        try:
            return self.model.objects.get(pk=self.pk)
        except self.model.DoesNotExist:
            raise AJAXError(404, _('%s with id of "%s" not found.') % (
                self.model.__name__, self.pk))
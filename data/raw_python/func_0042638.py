def get_fieldsets(self):
        """
        Hook for specifying fieldsets. If 'self.fieldsets' is
        empty this will default to include all the fields in
        the form with a title of None.
        """

        if self.fieldsets:
            return self.fieldsets
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        fields = form.base_fields.keys()

        readonly_fields = self.get_readonly_fields()
        if readonly_fields:
            fields.extend(readonly_fields)

        return [(None, {'fields': fields})]
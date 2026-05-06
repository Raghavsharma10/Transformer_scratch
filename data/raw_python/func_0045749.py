def clean(self, *args, **kwargs):
        """
        Potentially, these fields should validate against context-based
        queries.

        If a context variable has been transmitted to the field, it's being
        used to 'reset' the queryset and make sure the chosen item fits to
        the user context.
        """
        self.transmit_agnocomplete_context()
        return super(AgnocompleteMixin, self).clean(*args, **kwargs)
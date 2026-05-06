def get_qs(self):
        """
        Returns a mapping that will be used to generate
        the query string for the api url. Any values
        in the the `limit_choices_to` specified on the
        foreign key field and any arguments specified on
        self.extra_query_kwargs are converted to a format
        that can be used in a query string and returned as
        a dictionary.
        """
        qs = url_params_from_lookup_dict(self.rel.limit_choices_to)
        if not qs:
            qs = {}

        if self.extra_query_kwargs:
            qs.update(self.extra_query_kwargs)
        return qs
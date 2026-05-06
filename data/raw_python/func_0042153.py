def get_filter(self, **filter_kwargs):
        """
        Combines the Q objects returned by a valid
        filter form with any other arguments and
        returns a list of Q objects that can be passed
        to a queryset.
        """

        q_objects = super(ListView, self).get_filter(**filter_kwargs)
        form = self.get_filter_form()
        if form:
            q_objects.extend(form.get_filter())

        return q_objects
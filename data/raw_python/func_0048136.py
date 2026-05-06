def get_form_kwargs(self):
        """
        Return the form kwargs.

        This method injects the context variable, defined in
        :meth:`get_agnocomplete_context`. Override this method to adjust it to
        your needs.
        """
        data = super(UserContextFormViewMixin, self).get_form_kwargs()
        data.update({
            'user': self.get_agnocomplete_context(),
        })
        return data
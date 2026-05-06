def get_form_kwargs(self):
        """
        Returns the keyword arguments for instantiating the form.
        """
        kwargs = {'initial': self.get_initial(),
                  'instance': self.object}

        if self.request.method in ('POST', 'PUT') and self.can_submit:
            kwargs.update({
                'data': self.request.POST,
                'files': self.request.FILES
            })
        return kwargs
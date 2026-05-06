def get_form_kwargs(self):
        """
        Returns the keyword arguments for instantiating the form.

        :rtype: dict.
        """
        kwargs = {
            'initial':  self.get_initial(),
            'prefix':   self.get_prefix(),
        }

        #noinspection PyUnresolvedReferences
        if self.request.method in ('POST', 'PUT'):
            #noinspection PyUnresolvedReferences
            kwargs.update({
                'data':     self.request.POST,
                'files':    self.request.FILES,
            })

        if hasattr(self, 'object'):
            kwargs.update({'instance': self.object})

        return kwargs
def get_context_data(self, **kwargs):
        """ Add in the field to use for the name field """
        context = super(SmartDeleteView, self).get_context_data(**kwargs)
        context['name_field'] = self.name_field
        context['cancel_url'] = self.get_cancel_url()
        return context
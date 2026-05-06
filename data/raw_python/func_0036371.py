def get_context_data(self, **kwargs):
        """This add in the context of list_type and returns this as Alive."""
        
        context = super(AnimalListAlive, self).get_context_data(**kwargs)
        context['list_type'] = 'Alive'
        return context
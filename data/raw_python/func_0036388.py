def get_context_data(self, **kwargs):
        """This add in the context of list_type and returns this as whatever the crosstype was."""
        
        context = super(CrossTypeAnimalList, self).get_context_data(**kwargs)
        context['list_type'] = self.kwargs['breeding_type']
        return context
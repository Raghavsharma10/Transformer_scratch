def get_context_data(self, **kwargs):
        """This adds into the context of breeding_type and sets it to Active."""
        
        context = super(BreedingList, self).get_context_data(**kwargs)
        context['breeding_type'] = "Active" 
        return context
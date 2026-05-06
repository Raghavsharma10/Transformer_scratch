def get_context_data(self, **kwargs):
        """This add in the context of breeding_type and sets it to Search it also returns the query and the queryset."""
        query = self.request.GET.get('q', '')
        context = super(BreedingSearch, self).get_context_data(**kwargs)
        context['breeding_type'] = "Search"
        context['query'] = query
        if query:
            context['results'] = Breeding.objects.filter(Cage__icontains=query).distinct()
        else:
            context['results'] = []        
        return context
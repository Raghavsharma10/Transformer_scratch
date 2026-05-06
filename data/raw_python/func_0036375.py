def get_context_data(self, **kwargs):
        """This add in the context of strain_list_alive (which filters for all alive animals) and cages which filters for the number of current cages."""
        
        context = super(StrainList, self).get_context_data(**kwargs)
        context['strain_list_alive'] = Strain.objects.filter(animal__Alive=True).annotate(alive=Count('animal'))
        context['cages'] = Animal.objects.filter(Alive=True).values("Cage")        
        return context
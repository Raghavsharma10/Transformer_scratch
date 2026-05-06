def get_context_data(self, **kwargs):
        """This adds into the context of strain_list_all (which filters for all alive :class:`~mousedb.animal.models.Animal` objects and active cages) and cages which filters for the number of current cages."""
        
        strain = super(StrainDetail, self).get_object()
        context = super(StrainDetail, self).get_context_data(**kwargs)
        context['breeding_cages'] = Breeding.objects.filter(Strain=strain)
        context['animal_list'] = Animal.objects.filter(Strain=strain).order_by('Background','Genotype')
        context['cages'] = Animal.objects.filter(Strain=strain).values("Cage").distinct()
        context['active'] = False        
        return context
def queryset(self):
        """This function sets the queryset according to the keyword arguments.
        For the crosstype, the input value is the the display value of CROSS_TYPE.
        This is done because the spaces in HET vs HET are not recognized.  
        Therefore the queryset must be matched exactly (ie by case so Intercross not intercross). 
        The function also filters the strain by the strain_slug keyword argument.
        """
        
        from mousedb.animal.models import CROSS_TYPE    
        crosstype_reverse = dict((v, k) for k, v in CROSS_TYPE)
        try:
            crosstype = crosstype_reverse[self.kwargs['breeding_type']]
        except KeyError:
            raise Http404
        strain = get_object_or_404(Strain, Strain_slug=self.kwargs['strain_slug'])
        if strain:
            return Animal.objects.filter(Strain=strain,Breeding__Crosstype=crosstype)
        else:
            raise Http404
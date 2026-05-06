def get_queryset(self):
        '''The queryset is filtered by measurements of animals which are part of that strain.'''
        strain = get_object_or_404(Strain, Strain_slug=self.kwargs['strain_slug'])
        animals = Animal.objects.filter(Strain=strain)
        return Measurement.objects.filter(animal=animals)
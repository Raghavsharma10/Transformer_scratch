def get_queryset(self):
        '''The queryset is filtered by measurements of animals which are part of that strain.'''
        cohort = get_object_or_404(Cohort, slug=self.kwargs['slug'])
        animals = cohort.animals.all()
        return Measurement.objects.filter(animal=animals)
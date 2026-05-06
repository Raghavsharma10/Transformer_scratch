def get(self, request, *args, **kwargs):
        '''The queryset is filtered by measurements of animals which are part of that strain.'''
        cohort = get_object_or_404(Cohort, slug=self.kwargs['slug'])
        animals = cohort.animals.all()
        measurements = Measurement.objects.filter(animal=animals)    
        return data_csv(self.request, measurements)
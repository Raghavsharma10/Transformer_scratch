def get_queryset(self):
        """The queryset is over-ridden to show only plug events in which the strain matches the breeding strain."""
        self.strain = get_object_or_404(Strain, Strain_slug__iexact=self.kwargs['slug'])
        return PlugEvents.objects.filter(Breeding__Strain=self.strain)
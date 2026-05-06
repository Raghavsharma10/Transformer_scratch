def unweaned(self):
        """This attribute generates a queryset of unweaned animals for this breeding cage.  It is filtered for only Alive animals."""	
        return Animal.objects.filter(Breeding=self, Weaned__isnull=True, Alive=True)
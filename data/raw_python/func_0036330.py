def save(self):
        """The save function for a breeding cage has to automatic over-rides, Active and the Cage for the Breeder.
        
        In the case of Active, if an End field is specified, then the Active field is set to False.
        In the case of Cage, if a Cage is provided, and animals are specified under Male or Females for a Breeding object, then the Cage field for those animals is set to that of the breeding cage.  The same is true for both Rack and Rack Position."""
        if self.End:
            self.Active = False
        #if self.Cage:
        #    if self.Females:               
        #        for female_breeder in self.Females:
        #            female_breeder.Cage = self.Cage
        #            female_breeder.save()
        #    if self.Male:
        #        for male_breeder in self.Male:
        #            male_breeder.Cage = self.Cage
        #            male_breeder.save()
        super(Breeding, self).save()
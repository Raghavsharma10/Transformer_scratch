def save(self):
        """Over-rides the default save function for PlugEvents.

        If a sacrifice date is set for an object in this model, then Active is set to False."""
        if self.SacrificeDate:
            self.Active = False
        super(PlugEvents, self).save()
def save(self):
        """The save method for Animal class is over-ridden to set Alive=False when a Death date is entered.  This is not the case for a cause of death."""
        if self.Death:
            self.Alive = False
        super(Animal, self).save()
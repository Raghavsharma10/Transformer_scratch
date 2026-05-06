def age(self):
        """Calculates the animals age, relative to the current date (if alive) or the date of death (if not)."""
        if self.Death:
            age =  self.Death - self.Born
        else:    
            age =  datetime.date.today() - self.Born
        return age.days
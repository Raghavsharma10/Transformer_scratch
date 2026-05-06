def duration(self):
        """Calculates the breeding cage's duration.

        This is relative to the current date (if alive) or the date of inactivation (if not).
        The duration is formatted in days."""
        if self.End:
            age =  self.End - self.Start
        else:    
            age =  datetime.date.today() - self.Start
        return age.days
def _departure(self) -> datetime:
        """Extract departure time."""
        departure_time = datetime.strptime(
            self.journey.MainStop.BasicStop.Dep.Time.text, "%H:%M"
        ).time()
        if departure_time > (self.now - timedelta(hours=1)).time():
            return datetime.combine(self.now.date(), departure_time)
        return datetime.combine(self.now.date() + timedelta(days=1), departure_time)
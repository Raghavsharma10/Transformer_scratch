def zmanim(self):
        """Return a dictionary of the zmanim the object represents."""
        return {key: self.utc_minute_timezone(value) for
                key, value in self.get_utc_sun_time_full().items()}
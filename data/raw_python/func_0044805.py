def _havdalah_datetime(self):
        """Compute the havdalah time based on settings."""
        if self.havdalah_offset == 0:
            return self.zmanim["three_stars"]
        # Otherwise, use the offset.
        return (self.zmanim["sunset"]
                + dt.timedelta(minutes=self.havdalah_offset))
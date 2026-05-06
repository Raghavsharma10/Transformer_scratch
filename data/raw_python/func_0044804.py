def candle_lighting(self):
        """Return the time for candle lighting, or None if not applicable."""
        today = HDate(gdate=self.date, diaspora=self.location.diaspora)

        tomorrow = HDate(gdate=self.date + dt.timedelta(days=1),
                         diaspora=self.location.diaspora)

        # If today is a Yom Tov or Shabbat, and tomorrow is a Yom Tov or
        # Shabbat return the havdalah time as the candle lighting time.
        if ((today.is_yom_tov or today.is_shabbat)
                and (tomorrow.is_yom_tov or tomorrow.is_shabbat)):
            return self._havdalah_datetime

        # Otherwise, if today is Friday or erev Yom Tov, return candle
        # lighting.
        if tomorrow.is_shabbat or tomorrow.is_yom_tov:
            return (self.zmanim["sunset"]
                    - dt.timedelta(minutes=self.candle_lighting_offset))
        return None
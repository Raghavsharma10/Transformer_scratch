def get_utc_sun_time_full(self):
        """Return a list of Jewish times for the given location."""
        # sunset and rise time
        sunrise, sunset = self._get_utc_sun_time_deg(90.833)

        # shaa zmanit by gara, 1/12 of light time
        sun_hour = (sunset - sunrise) // 12
        midday = (sunset + sunrise) // 2

        # get times of the different sun angles
        first_light, _ = self._get_utc_sun_time_deg(106.1)
        talit, _ = self._get_utc_sun_time_deg(101.0)
        _, first_stars = self._get_utc_sun_time_deg(96.0)
        _, three_stars = self._get_utc_sun_time_deg(98.5)
        mga_sunhour = (midday - first_light) / 6

        res = dict(sunrise=sunrise, sunset=sunset, sun_hour=sun_hour,
                   midday=midday, first_light=first_light, talit=talit,
                   first_stars=first_stars, three_stars=three_stars,
                   plag_mincha=sunset - 1.25 * sun_hour,
                   stars_out=sunset + 18. * sun_hour / 60.,
                   small_mincha=sunrise + 9.5 * sun_hour,
                   big_mincha=sunrise + 6.5 * sun_hour,
                   mga_end_shma=first_light + mga_sunhour * 3.,
                   gra_end_shma=sunrise + sun_hour * 3.,
                   mga_end_tfila=first_light + mga_sunhour * 4.,
                   gra_end_tfila=sunrise + sun_hour * 4.,
                   midnight=midday + 12 * 60.)
        return res
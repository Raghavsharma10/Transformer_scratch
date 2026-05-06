def getMoonPhase(self):
        """ Returns the phase of the moon. """
        sun = self.getObject(const.SUN)
        moon = self.getObject(const.MOON)
        dist = angle.distance(sun.lon, moon.lon)
        if dist < 90:
            return const.MOON_FIRST_QUARTER
        elif dist < 180:
            return const.MOON_SECOND_QUARTER
        elif dist < 270:
            return const.MOON_THIRD_QUARTER
        else:
            return const.MOON_LAST_QUARTER
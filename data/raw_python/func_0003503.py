def earth_sun_d(dtime):
        """ Earth-sun distance in AU
        
        :param dtime time, e.g. datetime.datetime(2007, 5, 1)
        :type datetime object
        :return float(distance from sun to earth in astronomical units)
        """
        doy = int(dtime.strftime('%j'))
        rad_term = 0.9856 * (doy - 4) * pi / 180
        distance_au = 1 - 0.01672 * cos(rad_term)
        return distance_au
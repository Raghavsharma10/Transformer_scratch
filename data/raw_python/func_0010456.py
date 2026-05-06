def formatted(self):
    ''' print a nicely formatted output of this report '''

    return """
Weather Station: %s (%s, %s)
Elevation: %s m
Time: %s UTC
Air Temperature: %s C (%s F)
Wind Speed: %s m/s (%s mph)
Wind Direction: %s
Present Weather Obs: %s
Precipitation: %s
Cloud Coverage: %s oktas
Cloud Summation: %s
Solar Irradiance: %s 
    """ % (self.weather_station, self.latitude, self.longitude,
           self.elevation, self.datetime, self.air_temperature,
           self.air_temperature.get_fahrenheit(), self.wind_speed,
           self.wind_speed.get_miles(), self.wind_direction,
           str(self.present_weather), str(self.precipitation),
           str(self.sky_cover), str(self.sky_cover_summation),
           str(self.solar_irradiance))
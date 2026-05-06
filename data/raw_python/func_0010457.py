def loads(self, noaa_string):
    ''' load in a report (or set) from a string '''
    self.raw = noaa_string
    self.weather_station = noaa_string[4:10]
    self.wban = noaa_string[10:15]
    expected_length = int(noaa_string[0:4]) + self.PREAMBLE_LENGTH
    actual_length = len(noaa_string)
    if actual_length != expected_length:
      msg = "Non matching lengths. Expected %d, got %d" % (expected_length,
                                                           actual_length)
      raise ish_reportException(msg)

    try:
      self.datetime = datetime.strptime(noaa_string[15:27], '%Y%m%d%H%M')
    except ValueError:
      ''' some cases, we get 2400 hours, which is really the next day, so 
      this is a workaround for those cases '''
      time = noaa_string[15:27]
      time = time.replace("2400", "2300")
      self.datetime = datetime.strptime(time, '%Y%m%d%H%M')
      self.datetime += timedelta(hours=1)

    self.datetime = self.datetime.replace(tzinfo=pytz.UTC)

    self.report_type = ReportType(noaa_string[41:46].strip())

    self.latitude = float(noaa_string[28:34]) / self.GEO_SCALE
    self.longitude = float(noaa_string[34:41]) / self.GEO_SCALE
    self.elevation = int(noaa_string[46:51])

    ''' other mandatory fields '''
    self.wind_direction = Direction(noaa_string[60:63],
                                    Direction.RADIANS,
                                    noaa_string[63:64])
    self.wind_observation_direction_type = noaa_string[64:64]
    self.wind_speed = Speed(int(noaa_string[65:69]) / float(self.SPEED_SCALE),
                            Speed.METERSPERSECOND,
                            noaa_string[69:70])
    self.sky_ceiling = Distance(int(noaa_string[70:75]),
                                Distance.METERS,
                                noaa_string[75:76])
    self.sky_ceiling_determination = noaa_string[76:77]
    self.visibility_distance = Distance(int(noaa_string[78:84]),
                                        Distance.METERS,
                                        noaa_string[84:85]) 
    self.visibility_variability = noaa_string[85:86]
    self.visibility_variability_quality = noaa_string[86:87]

    self.air_temperature = Temperature(int(noaa_string[87:92]) / self.TEMPERATURE_SCALE,
                                           Units.CELSIUS,
                                           noaa_string[92:93])
    self.dew_point = Temperature(int(noaa_string[93:98]) / self.TEMPERATURE_SCALE,
                                 Units.CELSIUS,
                                 noaa_string[98:99])

    self.humidity = Humidity(str(self.air_temperature), str(self.dew_point))
    self.sea_level_pressure = Pressure(int(noaa_string[99:104])/self.PRESSURE_SCALE,
                                       Pressure.HECTOPASCALS,
                                       noaa_string[104:104])

    ''' handle the additional fields '''
    additional = noaa_string[105:108]
    if additional == 'ADD':
      position = 108
      while position < expected_length:
        try:
          (position, (addl_code, addl_string)) = self._get_component(noaa_string,
                                                                     position)
          self._additional[addl_code] = addl_string
        except ish_reportException as err:
          ''' this catches when we move to remarks section '''
          break

    ''' handle the remarks section if it exists '''
    try:
      position = noaa_string.index('REM', 108) 
      self._get_remarks_component(noaa_string, position)
    except (ish_reportException, ValueError) as err:
      ''' this catches when we move to EQD section '''

    return self
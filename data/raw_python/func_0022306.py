def read(self, vals):
        """Read values.

        Args:
            vals (list): list of strings representing values

        """
        i = 0
        if len(vals[i]) == 0:
            self.year = None
        else:
            self.year = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.month = None
        else:
            self.month = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.day = None
        else:
            self.day = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.hour = None
        else:
            self.hour = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.minute = None
        else:
            self.minute = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.data_source_and_uncertainty_flags = None
        else:
            self.data_source_and_uncertainty_flags = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.dry_bulb_temperature = None
        else:
            self.dry_bulb_temperature = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.dew_point_temperature = None
        else:
            self.dew_point_temperature = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.relative_humidity = None
        else:
            self.relative_humidity = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.atmospheric_station_pressure = None
        else:
            self.atmospheric_station_pressure = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.extraterrestrial_horizontal_radiation = None
        else:
            self.extraterrestrial_horizontal_radiation = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.extraterrestrial_direct_normal_radiation = None
        else:
            self.extraterrestrial_direct_normal_radiation = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.horizontal_infrared_radiation_intensity = None
        else:
            self.horizontal_infrared_radiation_intensity = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.global_horizontal_radiation = None
        else:
            self.global_horizontal_radiation = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.direct_normal_radiation = None
        else:
            self.direct_normal_radiation = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.diffuse_horizontal_radiation = None
        else:
            self.diffuse_horizontal_radiation = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.global_horizontal_illuminance = None
        else:
            self.global_horizontal_illuminance = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.direct_normal_illuminance = None
        else:
            self.direct_normal_illuminance = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.diffuse_horizontal_illuminance = None
        else:
            self.diffuse_horizontal_illuminance = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.zenith_luminance = None
        else:
            self.zenith_luminance = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.wind_direction = None
        else:
            self.wind_direction = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.wind_speed = None
        else:
            self.wind_speed = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.total_sky_cover = None
        else:
            self.total_sky_cover = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.opaque_sky_cover = None
        else:
            self.opaque_sky_cover = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.visibility = None
        else:
            self.visibility = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.ceiling_height = None
        else:
            self.ceiling_height = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.present_weather_observation = None
        else:
            self.present_weather_observation = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.present_weather_codes = None
        else:
            self.present_weather_codes = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.precipitable_water = None
        else:
            self.precipitable_water = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.aerosol_optical_depth = None
        else:
            self.aerosol_optical_depth = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.snow_depth = None
        else:
            self.snow_depth = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.days_since_last_snowfall = None
        else:
            self.days_since_last_snowfall = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.albedo = None
        else:
            self.albedo = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.liquid_precipitation_depth = None
        else:
            self.liquid_precipitation_depth = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.liquid_precipitation_quantity = None
        else:
            self.liquid_precipitation_quantity = vals[i]
        i += 1
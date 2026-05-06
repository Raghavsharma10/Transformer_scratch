def export(self, top=True):
        """Exports object to its string representation.

        Args:
            top (bool):  if True appends `internal_name` before values.
                All non list objects should be exported with value top=True,
                all list objects, that are embedded in as fields inlist objects
                should be exported with `top`=False

        Returns:
            str: The objects string representation

        """
        out = []
        if top:
            out.append(self._internal_name)
        out.append(self._to_str(self.year))
        out.append(self._to_str(self.month))
        out.append(self._to_str(self.day))
        out.append(self._to_str(self.hour))
        out.append(self._to_str(self.minute))
        out.append(self._to_str(self.data_source_and_uncertainty_flags))
        out.append(self._to_str(self.dry_bulb_temperature))
        out.append(self._to_str(self.dew_point_temperature))
        out.append(self._to_str(self.relative_humidity))
        out.append(self._to_str(self.atmospheric_station_pressure))
        out.append(self._to_str(self.extraterrestrial_horizontal_radiation))
        out.append(self._to_str(self.extraterrestrial_direct_normal_radiation))
        out.append(self._to_str(self.horizontal_infrared_radiation_intensity))
        out.append(self._to_str(self.global_horizontal_radiation))
        out.append(self._to_str(self.direct_normal_radiation))
        out.append(self._to_str(self.diffuse_horizontal_radiation))
        out.append(self._to_str(self.global_horizontal_illuminance))
        out.append(self._to_str(self.direct_normal_illuminance))
        out.append(self._to_str(self.diffuse_horizontal_illuminance))
        out.append(self._to_str(self.zenith_luminance))
        out.append(self._to_str(self.wind_direction))
        out.append(self._to_str(self.wind_speed))
        out.append(self._to_str(self.total_sky_cover))
        out.append(self._to_str(self.opaque_sky_cover))
        out.append(self._to_str(self.visibility))
        out.append(self._to_str(self.ceiling_height))
        out.append(self._to_str(self.present_weather_observation))
        out.append(self._to_str(self.present_weather_codes))
        out.append(self._to_str(self.precipitable_water))
        out.append(self._to_str(self.aerosol_optical_depth))
        out.append(self._to_str(self.snow_depth))
        out.append(self._to_str(self.days_since_last_snowfall))
        out.append(self._to_str(self.albedo))
        out.append(self._to_str(self.liquid_precipitation_depth))
        out.append(self._to_str(self.liquid_precipitation_quantity))
        return ",".join(out)
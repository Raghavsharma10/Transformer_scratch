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
        out.append(self._to_str(self.ground_temperature_depth))
        out.append(self._to_str(self.depth_soil_conductivity))
        out.append(self._to_str(self.depth_soil_density))
        out.append(self._to_str(self.depth_soil_specific_heat))
        out.append(self._to_str(self.depth_january_average_ground_temperature))
        out.append(
            self._to_str(
                self.depth_february_average_ground_temperature))
        out.append(self._to_str(self.depth_march_average_ground_temperature))
        out.append(self._to_str(self.depth_april_average_ground_temperature))
        out.append(self._to_str(self.depth_may_average_ground_temperature))
        out.append(self._to_str(self.depth_june_average_ground_temperature))
        out.append(self._to_str(self.depth_july_average_ground_temperature))
        out.append(self._to_str(self.depth_august_average_ground_temperature))
        out.append(
            self._to_str(
                self.depth_september_average_ground_temperature))
        out.append(self._to_str(self.depth_october_average_ground_temperature))
        out.append(
            self._to_str(
                self.depth_november_average_ground_temperature))
        out.append(
            self._to_str(
                self.depth_december_average_ground_temperature))
        return ",".join(out)
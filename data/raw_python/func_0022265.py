def read(self, vals):
        """Read values.

        Args:
            vals (list): list of strings representing values

        """
        i = 0
        if len(vals[i]) == 0:
            self.ground_temperature_depth = None
        else:
            self.ground_temperature_depth = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.depth_soil_conductivity = None
        else:
            self.depth_soil_conductivity = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.depth_soil_density = None
        else:
            self.depth_soil_density = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.depth_soil_specific_heat = None
        else:
            self.depth_soil_specific_heat = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.depth_january_average_ground_temperature = None
        else:
            self.depth_january_average_ground_temperature = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.depth_february_average_ground_temperature = None
        else:
            self.depth_february_average_ground_temperature = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.depth_march_average_ground_temperature = None
        else:
            self.depth_march_average_ground_temperature = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.depth_april_average_ground_temperature = None
        else:
            self.depth_april_average_ground_temperature = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.depth_may_average_ground_temperature = None
        else:
            self.depth_may_average_ground_temperature = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.depth_june_average_ground_temperature = None
        else:
            self.depth_june_average_ground_temperature = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.depth_july_average_ground_temperature = None
        else:
            self.depth_july_average_ground_temperature = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.depth_august_average_ground_temperature = None
        else:
            self.depth_august_average_ground_temperature = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.depth_september_average_ground_temperature = None
        else:
            self.depth_september_average_ground_temperature = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.depth_october_average_ground_temperature = None
        else:
            self.depth_october_average_ground_temperature = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.depth_november_average_ground_temperature = None
        else:
            self.depth_november_average_ground_temperature = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.depth_december_average_ground_temperature = None
        else:
            self.depth_december_average_ground_temperature = vals[i]
        i += 1
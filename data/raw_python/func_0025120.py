def format_grib_name(self, selected_variable):
        """
        Assigns name to grib2 message number with name 'unknown'. Names based on NOAA grib2 abbreviations.
        Args:
            selected_variable(str): name of selected variable for loading
        Names:
            3: LCDC: Low Cloud Cover
            4: MCDC: Medium Cloud Cover
            5: HCDC: High Cloud Cover
            197: RETOP: Echo Top
            198: MAXREF: Hourly Maximum of Simulated Reflectivity at 1 km AGL
            199: MXUPHL: Hourly Maximum of Updraft Helicity over Layer 2km to 5 km AGL, and 0km to 3km AGL
                    examples:' MXUPHL_5000' or 'MXUPHL_3000'
            200: MNUPHL: Hourly Minimum of Updraft Helicity at same levels of MXUPHL
                     examples:' MNUPHL_5000' or 'MNUPHL_3000'
            220: MAXUVV: Hourly Maximum of Upward Vertical Velocity in the lowest 400hPa
            221: MAXDVV: Hourly Maximum of Downward Vertical Velocity in the lowest 400hPa
            222: MAXUW: U Component of Hourly Maximum 10m Wind Speed
            223: MAXVW: V Component of Hourly Maximum 10m Wind Speed
        Returns:
            Given an uknown string name of a variable, returns the grib2 message Id
            and units of the variable, based on the self.unknown_name and
            self.unknown_units dictonaries above. Allows access of
            data values of unknown variable name, given the ID.
        """
        names = self.unknown_names
        units = self.unknown_units
        for key, value in names.items():
            if selected_variable == value:
                Id = key
                u = units[key]
        return Id, u
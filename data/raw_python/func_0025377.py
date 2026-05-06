def init_file(self, filename, time_units="seconds since 1970-01-01T00:00"):
        """
        Initializes netCDF file for writing

        Args:
            filename: Name of the netCDF file
            time_units: Units for the time variable in format "<time> since <date string>"
        Returns:
            Dataset object
        """
        if os.access(filename, os.R_OK):
            out_data = Dataset(filename, "r+")
        else:
            out_data = Dataset(filename, "w")
            if len(self.data.shape) == 2:
                for d, dim in enumerate(["y", "x"]):
                    out_data.createDimension(dim, self.data.shape[d])
            else:
                for d, dim in enumerate(["y", "x"]):
                    out_data.createDimension(dim, self.data.shape[d+1])
            out_data.createDimension("time", len(self.times))
            time_var = out_data.createVariable("time", "i8", ("time",))
            time_var[:] = date2num(self.times.to_pydatetime(), time_units)
            time_var.units = time_units
            out_data.Conventions = "CF-1.6"
        return out_data
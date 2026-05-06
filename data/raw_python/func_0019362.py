def write(self, timeunit, timepoints) -> None:
        """Open a new NetCDF file temporarily and call method
        |NetCDFVariableBase.write| of all handled |NetCDFVariableBase|
        objects."""
        with netcdf4.Dataset(self.filepath, "w") as ncfile:
            ncfile.Conventions = 'CF-1.6'
            self._insert_timepoints(ncfile, timepoints, timeunit)
            for variable in self.variables.values():
                variable.write(ncfile)
def read(self) -> None:
        """Open an existing NetCDF file temporarily and call method
        |NetCDFVariableDeep.read| of all handled |NetCDFVariableBase|
        objects."""
        try:
            with netcdf4.Dataset(self.filepath, "r") as ncfile:
                timegrid = query_timegrid(ncfile)
                for variable in self.variables.values():
                    variable.read(ncfile, timegrid)
        except BaseException:
            objecttools.augment_excmessage(
                f'While trying to read data from NetCDF file `{self.filepath}`')
def load_data(self):
        """
        Load data from netCDF file objects or list of netCDF file objects. Handles special variable name formats.

        Returns:
            Array of data loaded from files in (time, y, x) dimensions, Units
        """
        units = ""
        if self.file_objects[0] is None:
            raise IOError()
        var_name, z_index = self.format_var_name(self.variable, list(self.file_objects[0].variables.keys()))
        ntimes = 0
        if 'time' in self.file_objects[0].variables[var_name].dimensions:
            ntimes = len(self.file_objects[0].dimensions['time'])

        if ntimes > 1:
            if z_index is None:
                data = self.file_objects[0].variables[var_name][self.forecast_hours].astype(np.float32)
            else:
                data = self.file_objects[0].variables[var_name][self.forecast_hours, z_index].astype(np.float32)
        else:
            y_dim, x_dim = self.file_objects[0].variables[var_name].shape[-2:]
            data = np.zeros((len(self.valid_dates), y_dim, x_dim), dtype=np.float32)
            for f, file_object in enumerate(self.file_objects):
                if file_object is not None:
                    if z_index is None:
                        data[f] = file_object.variables[var_name][0]
                    else:
                        data[f] = file_object.variables[var_name][0, z_index]
        if hasattr(self.file_objects[0].variables[var_name], "units"):
            units = self.file_objects[0].variables[var_name].units
        return data, units
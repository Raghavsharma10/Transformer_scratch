def load_data_old(self):
        """
        Loads time series of 2D data grids from each opened file. The code 
        handles loading a full time series from one file or individual time steps
        from multiple files. Missing files are supported.
        """
        units = ""
        if len(self.file_objects) == 1 and self.file_objects[0] is not None:
            data = self.file_objects[0].variables[self.variable][self.forecast_hours]
            if hasattr(self.file_objects[0].variables[self.variable], "units"):
                units = self.file_objects[0].variables[self.variable].units
        elif len(self.file_objects) > 1:
            grid_shape = [len(self.file_objects), 1, 1]
            for file_object in self.file_objects:
                if file_object is not None:
                    if self.variable in file_object.variables.keys():
                        grid_shape = file_object.variables[self.variable].shape
                    elif self.variable.ljust(6, "_") in file_object.variables.keys():
                        grid_shape = file_object.variables[self.variable.ljust(6, "_")].shape

                    else:
                        print("{0} not found".format(self.variable))
                        raise KeyError
                    break
            data = np.zeros((len(self.file_objects), grid_shape[1], grid_shape[2]))
            for f, file_object in enumerate(self.file_objects):
                if file_object is not None:
                    if self.variable in file_object.variables.keys():
                        var_name = self.variable
                    elif self.variable.ljust(6, "_") in file_object.variables.keys():
                        var_name = self.variable.ljust(6, "_")
                    else:
                        print("{0} not found".format(self.variable))
                        raise KeyError
                    data[f] = file_object.variables[var_name][0]
                    if units == "" and hasattr(file_object.variables[var_name], "units"):
                        units = file_object.variables[var_name].units
        else:
            data = None
        return data, units
def interpolate_to_netcdf(self, in_lon, in_lat, out_path, date_unit="seconds since 1970-01-01T00:00",
                              interp_type="spline"):
        """
        Calls the interpolation function and then saves the MRMS data to a netCDF file. It will also create 
        separate directories for each variable if they are not already available.
        """
        if interp_type == "spline":
            out_data = self.interpolate_grid(in_lon, in_lat)
        else:
            out_data = self.max_neighbor(in_lon, in_lat)
        if not os.access(out_path + self.variable, os.R_OK):
            try:
                os.mkdir(out_path + self.variable)
            except OSError:
                print(out_path + self.variable + " already created")
        out_file = out_path + self.variable + "/" + "{0}_{1}_{2}.nc".format(self.variable,
                                                                            self.start_date.strftime("%Y%m%d-%H:%M"),
                                                                            self.end_date.strftime("%Y%m%d-%H:%M"))
        out_obj = Dataset(out_file, "w")
        out_obj.createDimension("time", out_data.shape[0])
        out_obj.createDimension("y", out_data.shape[1])
        out_obj.createDimension("x", out_data.shape[2])
        data_var = out_obj.createVariable(self.variable, "f4", ("time", "y", "x"), zlib=True, 
                                          fill_value=-9999.0,
                                          least_significant_digit=3)
        data_var[:] = out_data
        data_var.long_name = self.variable
        data_var.coordinates = "latitude longitude"
        if "MESH" in self.variable or "QPE" in self.variable:
            data_var.units = "mm"
        elif "Reflectivity" in self.variable:
            data_var.units = "dBZ"
        elif "Rotation" in self.variable:
            data_var.units = "s-1"
        else:
            data_var.units = ""
        out_lon = out_obj.createVariable("longitude", "f4", ("y", "x"), zlib=True)
        out_lon[:] = in_lon
        out_lon.units = "degrees_east"
        out_lat = out_obj.createVariable("latitude", "f4", ("y", "x"), zlib=True)
        out_lat[:] = in_lat
        out_lat.units = "degrees_north"
        dates = out_obj.createVariable("time", "i8", ("time",), zlib=True)
        dates[:] = np.round(date2num(self.all_dates.to_pydatetime(), date_unit)).astype(np.int64)
        dates.long_name = "Valid date"
        dates.units = date_unit
        out_obj.Conventions="CF-1.6"
        out_obj.close()
        return
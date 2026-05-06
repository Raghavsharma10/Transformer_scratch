def load_coordinates(self):
        """
        Loads lat-lon coordinates from a netCDF file.
        """
        coord_file = Dataset(self.coordinate_file)
        if "lon" in coord_file.variables.keys():
            self.coordinates["lon"] = coord_file.variables["lon"][:]
            self.coordinates["lat"] = coord_file.variables["lat"][:]
        else:
            self.coordinates["lon"] = coord_file.variables["XLONG"][0]
            self.coordinates["lat"] = coord_file.variables["XLAT"][0]
        coord_file.close()
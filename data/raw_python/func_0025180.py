def load_map_info(self, map_file):
        """
        Load map projection information and create latitude, longitude, x, y, i, and j grids for the projection.

        Args:
            map_file: File specifying the projection information.
        """
        if self.ensemble_name.upper() == "SSEF":
            proj_dict, grid_dict = read_arps_map_file(map_file)
            self.dx = int(grid_dict["dx"])
            mapping_data = make_proj_grids(proj_dict, grid_dict)
            for m, v in mapping_data.items():
                setattr(self, m, v)
            self.i, self.j = np.indices(self.lon.shape)
            self.proj = get_proj_obj(proj_dict)
        elif self.ensemble_name.upper() in ["NCAR", "NCARSTORM", "HRRR", "VSE", "HREFV2"]:
            proj_dict, grid_dict = read_ncar_map_file(map_file)
            if self.member_name[0:7] == "1km_pbl": # Don't just look at the first 3 characters. You have to differentiate '1km_pbl1' and '1km_on_3km_pbl1'
                grid_dict["dx"] = 1000
                grid_dict["dy"] = 1000
                grid_dict["sw_lon"] = 258.697
                grid_dict["sw_lat"] = 23.999
                grid_dict["ne_lon"] = 282.868269206236
                grid_dict["ne_lat"] = 36.4822338520542 

            self.dx = int(grid_dict["dx"])
            mapping_data = make_proj_grids(proj_dict, grid_dict)
            for m, v in mapping_data.items():
                setattr(self, m, v)
            self.i, self.j = np.indices(self.lon.shape)
            self.proj = get_proj_obj(proj_dict)
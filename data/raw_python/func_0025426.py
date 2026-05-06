def extract_model_attributes(self, tracked_model_objects, storm_variables, potential_variables,
                                 tendency_variables=None, future_variables=None):
        """
        Extract model attribute data for each model track. Storm variables are those that describe the model storm
        directly, such as radar reflectivity or updraft helicity. Potential variables describe the surrounding
        environmental conditions of the storm, and should be extracted from the timestep before the storm arrives to
        reduce the chance of the storm contaminating the environmental values. Examples of potential variables include
        CAPE, shear, temperature, and dewpoint. Future variables are fields that occur in the hour after the extracted
        field.

        Args:
            tracked_model_objects: List of STObjects describing each forecasted storm
            storm_variables: List of storm variable names
            potential_variables: List of potential variable names.
            tendency_variables: List of tendency variables
        """
        if tendency_variables is None:
            tendency_variables = []
        if future_variables is None:
            future_variables = []
        model_grids = {}
        for l_var in ["lon", "lat"]:
            for model_obj in tracked_model_objects:
                model_obj.extract_attribute_array(getattr(self.model_grid, l_var), l_var)
        for storm_var in storm_variables:
            print("Storm {0} {1} {2}".format(storm_var,self.ensemble_member, self.run_date.strftime("%Y%m%d")))
            model_grids[storm_var] = ModelOutput(self.ensemble_name, self.ensemble_member,
                                                 self.run_date, storm_var, self.start_date - timedelta(hours=1),
                                                 self.end_date + timedelta(hours=1),
                                                 self.model_path,self.model_map_file, 
                                                 self.sector_ind_path,self.single_step)
            model_grids[storm_var].load_data()
            for model_obj in tracked_model_objects:
                model_obj.extract_attribute_grid(model_grids[storm_var])
            if storm_var not in potential_variables + tendency_variables + future_variables:
                del model_grids[storm_var]
        for potential_var in potential_variables:
            print("Potential {0} {1} {2}".format(potential_var,self.ensemble_member, self.run_date.strftime("%Y%m%d")))
            if potential_var not in model_grids.keys():
                model_grids[potential_var] = ModelOutput(self.ensemble_name, self.ensemble_member,
                                                         self.run_date, potential_var,
                                                         self.start_date - timedelta(hours=1),
                                                         self.end_date + timedelta(hours=1),
                                                         self.model_path, self.model_map_file, 
                                                         self.sector_ind_path,self.single_step)
                model_grids[potential_var].load_data()
            for model_obj in tracked_model_objects:
                model_obj.extract_attribute_grid(model_grids[potential_var], potential=True)
            if potential_var not in tendency_variables + future_variables:
                del model_grids[potential_var]
        for future_var in future_variables:
            print("Future {0} {1} {2}".format(future_var, self.ensemble_member, self.run_date.strftime("%Y%m%d")))
            if future_var not in model_grids.keys():
                model_grids[future_var] = ModelOutput(self.ensemble_name, self.ensemble_member,
                                                         self.run_date, future_var,
                                                         self.start_date - timedelta(hours=1),
                                                         self.end_date + timedelta(hours=1),
                                                         self.model_path, self.model_map_file,
                                                         self.sector_ind_path,self.single_step)
                model_grids[future_var].load_data()
            for model_obj in tracked_model_objects:
                model_obj.extract_attribute_grid(model_grids[future_var], future=True)
            if future_var not in tendency_variables:
                del model_grids[future_var]
        for tendency_var in tendency_variables:
            print("Tendency {0} {1} {2}".format(tendency_var, self.ensemble_member, self.run_date.strftime("%Y%m%d")))
            if tendency_var not in model_grids.keys():
                model_grids[tendency_var] = ModelOutput(self.ensemble_name, self.ensemble_member,
                                                        self.run_date, tendency_var,
                                                        self.start_date - timedelta(hours=1),
                                                        self.end_date,
                                                        self.model_path, self.model_map_file, 
                                                        self.sector_ind_path,self.single_step)
            for model_obj in tracked_model_objects:
                model_obj.extract_tendency_grid(model_grids[tendency_var])
            del model_grids[tendency_var]
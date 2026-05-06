def load_data(self):
        """
        Load the specified variable from the ensemble files, then close the files.
        """
        if self.ensemble_name.upper() == "SSEF":
            if self.variable[0:2] == "rh":
                pressure_level = self.variable[2:]
                relh_vars = ["sph", "tmp"]
                relh_vals = {}
                for var in relh_vars:
                    mg = SSEFModelGrid(self.member_name,
                                       self.run_date,
                                       var + pressure_level,
                                       self.start_date,
                                       self.end_date,
                                       self.path,
                                       single_step=self.single_step)
                    relh_vals[var], units = mg.load_data()
                    mg.close()
                self.data = relative_humidity_pressure_level(relh_vals["tmp"],
                                                             relh_vals["sph"],
                                                             float(pressure_level) * 100)
                self.units = "%"
            elif self.variable == "melth":
                input_vars = ["hgtsfc", "hgt700", "hgt500", "tmp700", "tmp500"]
                input_vals = {}
                for var in input_vars:
                    mg = SSEFModelGrid(self.member_name,
                                       self.run_date,
                                       var,
                                       self.start_date,
                                       self.end_date,
                                       self.path,
                                       single_step=self.single_step)
                    input_vals[var], units = mg.load_data()
                    mg.close()
                self.data = melting_layer_height(input_vals["hgtsfc"],
                                                 input_vals["hgt700"],
                                                 input_vals["hgt500"],
                                                 input_vals["tmp700"],
                                                 input_vals["tmp500"])
                self.units = "m"
            else:
                mg = SSEFModelGrid(self.member_name,
                                   self.run_date,
                                   self.variable,
                                   self.start_date,
                                   self.end_date,
                                   self.path,
                                   single_step=self.single_step)
                self.data, self.units = mg.load_data()
                mg.close()
        elif self.ensemble_name.upper() == "NCAR":
            mg = NCARModelGrid(self.member_name,
                               self.run_date,
                               self.variable,
                               self.start_date,
                               self.end_date,
                               self.path,
                               single_step=self.single_step)
            self.data, self.units = mg.load_data()
            mg.close()
        elif self.ensemble_name.upper() == "HREFV2":
            proj_dict, grid_dict = read_ncar_map_file(self.map_file)
            mapping_data = make_proj_grids(proj_dict, grid_dict)            
           
            mg = HREFv2ModelGrid(self.member_name,
                               self.run_date,
                               self.variable,
                               self.start_date,
                               self.end_date,
                               self.path,
                               mapping_data,
                               self.sector_ind_path,
                               single_step=self.single_step)

            self.data, self.units = mg.load_data()

        elif self.ensemble_name.upper() == "VSE":
            mg = VSEModelGrid(self.member_name,
                               self.run_date,
                               self.variable,
                               self.start_date,
                               self.end_date,
                               self.path,
                               single_step=self.single_step)
            self.data, self.units = mg.load_data()
            mg.close()
        elif self.ensemble_name.upper() == "HRRR":
            mg = HRRRModelGrid(self.run_date,
                               self.variable,
                               self.start_date,
                               self.end_date,
                               self.path)
            self.data, self.units = mg.load_data()
            mg.close()
        elif self.ensemble_name.upper() == "NCARSTORM":
            mg = NCARStormEventModelGrid(self.run_date,
                                         self.variable,
                                         self.start_date,
                                         self.end_date,
                                         self.path)
            self.data, self.units = mg.load_data()
            mg.close()
        else:
            print(self.ensemble_name + " not supported.")
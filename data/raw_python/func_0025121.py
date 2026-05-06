def load_data(self):
        """
            Loads data from grib2 file objects or list of grib2 file objects. Handles specific grib2 variable names
            and grib2 message numbers.
            Returns:
                    Array of data loaded from files in (time, y, x) dimensions, Units
        """
        file_objects = self.file_objects
        var = self.variable
        valid_date = self.valid_dates
        data = self.data
        unknown_names = self.unknown_names
        unknown_units = self.unknown_units
        member = self.member
        lat = self.lat
        lon = self.lon
      
        if self.sector_ind_path:
            inds_file = pd.read_csv(self.sector_ind_path+'sector_data_indices.csv') 
            inds = inds_file.loc[:,'indices']  
        out_x = self.mapping_data["x"]
        
        if not file_objects:
            print()
            print("No {0} model runs on {1}".format(member,self.run_date))
            print()
            units = None
            return self.data, units

    
        for f, file in enumerate(file_objects):
            grib = pygrib.open(file)
            if type(var) is int:
                data_values = grib[var].values
                #lat, lon = grib[var].latlons()
                #proj = Proj(grib[var].projparams)
                if grib[var].units == 'unknown':
                    Id = grib[var].parameterNumber
                    units = self.unknown_units[Id] 
                else:
                    units = grib[var].units
            elif type(var) is str:
                if '_' in var:
                    variable = var.split('_')[0]
                    level = int(var.split('_')[1])
                    if variable in unknown_names.values():
                        Id, units = self.format_grib_name(variable)
                        data_values = grib.select(parameterNumber=Id, level=level)[0].values
                        #lat, lon =  grib.select(parameterNumber=Id, level=level)[0].latlons()
                        #proj = Proj(grib.select(parameterNumber=Id, level=level)[0].projparams)

                    else:
                        data_values = grib.select(name=variable, level=level)[0].values
                        units = grib.select(name=variable, level=level)[0].units
                        #lat, lon  = grib.select(name=variable, level=level)[0].latlons()
                        #proj = Proj(grib.select(name=variable, level=level)[0].projparams)
                else:
                    if var in unknown_names.values():
                        Id, units = self.format_grib_name(var)
                        data_values = grib.select(parameterNumber=Id)[0].values
                        #lat, lon = grib.select(parameterNumber=Id)[0].latlons() 
                        #proj = Proj(grib.select(parameterNumber=Id)[0].projparams)

                    elif len(grib.select(name=var)) > 1:
                        raise NameError("Multiple '{0}' records found. Rename with level:'{0}_level'".format(var))

                    else:
                        data_values = grib.select(name=var)[0].values
                        units = grib.select(name=var)[0].units
                        #lat, lon = grib.select(name=var)[0].latlons()
                        #proj = Proj(grib.select(name=var)[0].projparams)

            if data is None:
                data = np.empty((len(valid_date), out_x.shape[0], out_x.shape[1]), dtype=float)
                if self.sector_ind_path:
                    data[f] = data_values[:].flatten()[inds].reshape(out_x.shape)
                else:
                    data[f]=data_values[:]
            else:
                if self.sector_ind_path:
                    data[f] = data_values[:].flatten()[inds].reshape(out_x.shape)
                else:
                    data[f]=data_values[:]
        
        return data, units
def fit_condition_models(self, model_names,
                             model_objs,
                             input_columns,
                             output_column="Matched",
                             output_threshold=0.0):
        """
        Fit machine learning models to predict whether or not hail will occur.
        Args:
            model_names: List of strings with the names for the particular machine learning models
            model_objs: scikit-learn style machine learning model objects.
            input_columns: list of the names of the columns used as input for the machine learning model
            output_column: name of the column used for labeling whether or not the event occurs
            output_threshold: splitting threshold to determine if event has occurred. Default 0.0
        """
        print("Fitting condition models")
        groups = self.data["train"]["member"][self.group_col].unique()
        
        weights = None

        for group in groups:
            print(group)
            group_data = self.data["train"]["combo"].loc[self.data["train"]["combo"][self.group_col] == group] 
            if self.sector:
        
                lon_obj = data.loc[:,'Centroid_Lon']
                lat_obj = data.loc[:,'Centroid_Lat']

                left_lon,right_lon = self.grid_dict["sw_lon"],self.grid_dict["ne_lon"]
                lower_lat,upper_lat = self.grid_dict["sw_lat"],self.grid_dict["ne_lat"]

                weights = np.where((left_lon<=lon_obj)&(right_lon>=lon_obj) &\
                    (lower_lat<=lat_obj)&(upper_lat>=lat_obj),1,0.3)
            
            output_data = np.where(group_data[output_column] > output_threshold, 1, 0)
            print("Ones: ", np.count_nonzero(output_data > 0), "Zeros: ", np.count_nonzero(output_data == 0))
            self.condition_models[group] = {}
            for m, model_name in enumerate(model_names):
                print(model_name)
                self.condition_models[group][model_name] = deepcopy(model_objs[m])
                try:
                    self.condition_models[group][model_name].fit(group_data[input_columns], 
                                output_data,sample_weight=weights)
                except:
                    self.condition_models[group][model_name].fit(group_data[input_columns], output_data)

                if hasattr(self.condition_models[group][model_name], "best_estimator_"):
                    print(self.condition_models[group][model_name].best_estimator_)
                    print(self.condition_models[group][model_name].best_score_)
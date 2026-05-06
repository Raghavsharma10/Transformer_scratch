def fit_size_distribution_models(self, model_names, model_objs, input_columns,
                                     output_columns=None, calibrate=False):
        """
        Fits multitask machine learning models to predict the parameters of a size distribution
        Args:
            model_names: List of machine learning model names
            model_objs: scikit-learn style machine learning model objects
            input_columns: Training data columns used as input for ML model
            output_columns: Training data columns used for prediction
            calibrate: Whether or not to fit a log-linear regression to predictions from ML model
        """
        if output_columns is None:
            output_columns = ["Shape", "Location", "Scale"]
        groups = np.unique(self.data["train"]["member"][self.group_col])
        
        weights=None
        
        for group in groups:
            group_data = self.data["train"]["combo"].loc[self.data["train"]["combo"][self.group_col] == group]
            group_data = group_data.dropna()
            group_data = group_data[group_data[output_columns[-1]] > 0]
            if self.sector:
        
                lon_obj = group_data.loc[:,'Centroid_Lon']
                lat_obj = group_data.loc[:,'Centroid_Lat']
                
                conus_lat_lon_points = zip(lon_obj.values.ravel(),lat_obj.values.ravel())
                center_lon, center_lat = self.proj_dict["lon_0"],self.proj_dict["lat_0"] 
            
                distances = np.array([np.sqrt((x-center_lon)**2+\
                        (y-center_lat)**2) for (x, y) in conus_lat_lon_points])
            
                min_dist, max_minus_min = min(distances),max(distances)-min(distances)

                distance_0_1 = [1.0-((d - min_dist)/(max_minus_min)) for d in distances]
                weights = np.array(distance_0_1)

            self.size_distribution_models[group] = {"multi": {}, "lognorm": {}}
            if calibrate:
                self.size_distribution_models[group]["calshape"] = {}
                self.size_distribution_models[group]["calscale"] = {}
            log_labels = np.log(group_data[output_columns].values)
            log_means = log_labels.mean(axis=0)
            log_sds = log_labels.std(axis=0)
            self.size_distribution_models[group]['lognorm']['mean'] = log_means
            self.size_distribution_models[group]['lognorm']['sd'] = log_sds
            for m, model_name in enumerate(model_names):
                print(group, model_name)
                self.size_distribution_models[group]["multi"][model_name] = deepcopy(model_objs[m])
                try:
                    self.size_distribution_models[group]["multi"][model_name].fit(group_data[input_columns],
                                                                              (log_labels - log_means) / log_sds,
                                                                        sample_weight=weights)
                except:
                    self.size_distribution_models[group]["multi"][model_name].fit(group_data[input_columns],
                                                                              (log_labels - log_means) / log_sds)
                if calibrate:
                    training_predictions = self.size_distribution_models[
                        group]["multi"][model_name].predict(group_data[input_columns])
                    self.size_distribution_models[group]["calshape"][model_name] = LinearRegression()
                    self.size_distribution_models[group]["calshape"][model_name].fit(training_predictions[:, 0:1],
                                                                                     (log_labels[:, 0] - log_means[0]) /
                                                                                     log_sds[
                                                                                         0],
                                                                                    sample_weight=weights)
                    self.size_distribution_models[group]["calscale"][model_name] = LinearRegression()
                    self.size_distribution_models[group]["calscale"][model_name].fit(training_predictions[:, 1:],
                                                                                     (log_labels[:, 1] - log_means[1]) /
                                                                                     log_sds[
                                                                                         1],
                                                                            sample_weight=weights)
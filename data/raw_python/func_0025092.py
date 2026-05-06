def fit_size_distribution_component_models(self, model_names, model_objs, input_columns, output_columns):
        """
        This calculates 2 principal components for the hail size distribution between the shape and scale parameters.
        Separate machine learning models are fit to predict each component.

        Args:
            model_names: List of machine learning model names
            model_objs: List of machine learning model objects.
            input_columns: List of input variables
            output_columns: Output columns, should contain Shape and Scale.

        Returns:

        """
        groups = np.unique(self.data["train"]["member"][self.group_col])
        
        weights=None

        for group in groups:
            print(group)
            group_data = self.data["train"]["combo"].loc[self.data["train"]["combo"][self.group_col] == group]
            group_data = group_data.dropna()
            group_data = group_data.loc[group_data[output_columns[-1]] > 0]
            
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

            
            self.size_distribution_models[group] = {"lognorm": {}}
            self.size_distribution_models[group]["lognorm"]["pca"] = PCA(n_components=len(output_columns))
            log_labels = np.log(group_data[output_columns].values)
            log_labels[:, np.where(output_columns == "Shape")[0]] *= -1
            log_means = log_labels.mean(axis=0)
            log_sds = log_labels.std(axis=0)
            log_norm_labels = (log_labels - log_means) / log_sds
            out_pc_labels = self.size_distribution_models[group]["lognorm"]["pca"].fit_transform(log_norm_labels)
            self.size_distribution_models[group]['lognorm']['mean'] = log_means
            self.size_distribution_models[group]['lognorm']['sd'] = log_sds
            for comp in range(len(output_columns)):
                self.size_distribution_models[group]["pc_{0:d}".format(comp)] = dict()
                for m, model_name in enumerate(model_names):
                    print(model_name, comp)
                    self.size_distribution_models[group][
                        "pc_{0:d}".format(comp)][model_name] = deepcopy(model_objs[m])
                    try:
                        self.size_distribution_models[group][
                            "pc_{0:d}".format(comp)][model_name].fit(group_data[input_columns],
                                                                 out_pc_labels[:, comp],
                                                            sample_weight=weights)
                    except:
                        self.size_distribution_models[group][
                            "pc_{0:d}".format(comp)][model_name].fit(group_data[input_columns],
                                                                 out_pc_labels[:, comp])
        return
def fit_condition_threshold_models(self, model_names, model_objs, input_columns, output_column="Matched",
                                       output_threshold=0.5, num_folds=5, threshold_score="ets"):
        """
        Fit models to predict hail/no-hail and use cross-validation to determine the probaility threshold that
        maximizes a skill score.

        Args:
            model_names: List of machine learning model names
            model_objs: List of Scikit-learn ML models
            input_columns: List of input variables in the training data
            output_column: Column used for prediction
            output_threshold: Values exceeding this threshold are considered positive events; below are nulls
            num_folds: Number of folds in the cross-validation procedure
            threshold_score: Score available in ContingencyTable used for determining the best probability threshold
        Returns:
            None
        """
        print("Fitting condition models")
        groups = self.data["train"]["member"][self.group_col].unique()
        
        weights=None

        for group in groups:
            print(group)
            group_data = self.data["train"]["combo"].iloc[
                np.where(self.data["train"]["combo"][self.group_col] == group)[0]]
            
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
        
            output_data = np.where(group_data.loc[:, output_column] > output_threshold, 1, 0)
            ones = np.count_nonzero(output_data > 0)
            print("Ones: ", ones, "Zeros: ", np.count_nonzero(output_data == 0))
            self.condition_models[group] = {}
            num_elements = group_data[input_columns].shape[0]
            
            for m, model_name in enumerate(model_names):
                print(model_name)    
                roc = DistributedROC(thresholds=np.arange(0, 1.1, 0.01))
                self.condition_models[group][model_name] = deepcopy(model_objs[m])

                try:
                    kf = KFold(n_splits=num_folds)
                    for train_index, test_index in kf.split(group_data[input_columns].values):
                        if np.count_nonzero(output_data[train_index]) > 0:
                            try:
                                self.condition_models[group][model_name].fit(
                                        group_data.iloc[train_index][input_columns],
                                        output_data[train_index],sample_weight=weights[train_index])
                            except:
                                self.condition_models[group][model_name].fit(
                                        group_data.iloc[train_index][input_columns],
                                        output_data[train_index])
                            
                            cv_preds = self.condition_models[group][model_name].predict_proba(
                                group_data.iloc[test_index][input_columns])[:,1]
                            
                            roc.update(cv_preds, output_data[test_index])
                        
                        else:
                            continue

                except TypeError:
                    kf = KFold(num_elements,n_folds=num_folds)
                    for train_index, test_index in kf:

                        if np.count_nonzero(output_data[train_index]) > 0:
                            try:
                                self.condition_models[group][model_name].fit(
                                        group_data.iloc[train_index][input_columns],
                                        output_data[train_index],sample_weight=weights[train_index])
                            except:
                                self.condition_models[group][model_name].fit(
                                        group_data.iloc[train_index][input_columns],
                                        output_data[train_index])
                            cv_preds = self.condition_models[group][model_name].predict_proba(
                                group_data.iloc[test_index][input_columns])[:, 1]
                            
                            roc.update(cv_preds, output_data[test_index])
                        
                        else:
                            continue

                self.condition_models[group][
                    model_name + "_condition_threshold"], _ = roc.max_threshold_score(threshold_score)
                print(model_name + " condition threshold: {0:0.3f}".format(
                self.condition_models[group][model_name + "_condition_threshold"]))
                self.condition_models[group][model_name].fit(group_data[input_columns], output_data)
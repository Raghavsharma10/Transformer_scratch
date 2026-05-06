def predict_size_distribution_models(self, model_names, input_columns, metadata_cols,
                                         data_mode="forecast", location=6, calibrate=False):
        """
        Make predictions using fitted size distribution models.
        Args:
            model_names: Name of the models for predictions
            input_columns: Data columns used for input into ML models
            metadata_cols: Columns from input data that should be included in the data frame with the predictions.
            data_mode: Set of data used as input for prediction models
            location: Value of fixed location parameter
            calibrate: Whether or not to apply calibration model
        Returns:
            Predictions in dictionary of data frames grouped by group type
        """
        groups = self.size_distribution_models.keys()
        predictions = {}
        for group in groups:
            group_data = self.data[data_mode]["combo"].loc[self.data[data_mode]["combo"][self.group_col] == group]
            predictions[group] = group_data[metadata_cols]
            if group_data.shape[0] > 0:
                log_mean = self.size_distribution_models[group]["lognorm"]["mean"]
                log_sd = self.size_distribution_models[group]["lognorm"]["sd"]
                for m, model_name in enumerate(model_names):
                    multi_predictions = self.size_distribution_models[group]["multi"][model_name].predict(
                        group_data[input_columns])
                    if calibrate:
                        multi_predictions[:, 0] = self.size_distribution_models[group]["calshape"][model_name].predict(
                            multi_predictions[:, 0:1])
                        multi_predictions[:, 1] = self.size_distribution_models[group]["calscale"][model_name].predict(
                            multi_predictions[:, 1:])
                    multi_predictions = np.exp(multi_predictions * log_sd + log_mean)
                    if multi_predictions.shape[1] == 2:
                        multi_predictions_temp = np.zeros((multi_predictions.shape[0], 3))
                        multi_predictions_temp[:, 0] = multi_predictions[:, 0]
                        multi_predictions_temp[:, 1] = location
                        multi_predictions_temp[:, 2] = multi_predictions[:, 1]
                        multi_predictions = multi_predictions_temp
                    for p, pred_col in enumerate(["shape", "location", "scale"]):
                        predictions[group][model_name].loc[:, model_name.replace(" ", "-") + "_" + pred_col] = \
                            multi_predictions[:, p]
        return predictions
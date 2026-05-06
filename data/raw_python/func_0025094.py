def predict_size_distribution_component_models(self, model_names, input_columns, output_columns, metadata_cols,
                                                   data_mode="forecast", location=6):
        """
        Make predictions using fitted size distribution models.
        Args:
            model_names: Name of the models for predictions
            input_columns: Data columns used for input into ML models
            output_columns: Names of output columns
            metadata_cols: Columns from input data that should be included in the data frame with the predictions.
            data_mode: Set of data used as input for prediction models
            location: Value of fixed location parameter
        Returns:
            Predictions in dictionary of data frames grouped by group type
        """
        groups = self.size_distribution_models.keys()
        predictions = pd.DataFrame(self.data[data_mode]["combo"][metadata_cols])
        for group in groups:
            group_idxs = self.data[data_mode]["combo"][self.group_col] == group
            group_count = np.count_nonzero(group_idxs)
            print(self.size_distribution_models[group])
            if group_count > 0:
                log_mean = self.size_distribution_models[group]["lognorm"]["mean"]
                log_sd = self.size_distribution_models[group]["lognorm"]["sd"]
                for m, model_name in enumerate(model_names):
                    raw_preds = np.zeros((group_count, len(output_columns)))
                    for c in range(len(output_columns)):
                        raw_preds[:, c] = self.size_distribution_models[group][
                            "pc_{0:d}".format(c)][model_name].predict(self.data[data_mode]["combo"].loc[group_idxs,
                                                                                                        input_columns])
                    log_norm_preds = self.size_distribution_models[group]["lognorm"]["pca"].inverse_transform(raw_preds)
                    log_norm_preds[:, 0] *= -1
                    multi_predictions = np.exp(log_norm_preds * log_sd + log_mean)
                    if multi_predictions.shape[1] == 2:
                        multi_predictions_temp = np.zeros((multi_predictions.shape[0], 3))
                        multi_predictions_temp[:, 0] = multi_predictions[:, 0]
                        multi_predictions_temp[:, 1] = location
                        multi_predictions_temp[:, 2] = multi_predictions[:, 1]
                        multi_predictions = multi_predictions_temp
                    for p, pred_col in enumerate(["shape", "location", "scale"]):
                        predictions.loc[group_idxs, model_name.replace(" ", "-") + "_" + pred_col] = \
                            multi_predictions[:, p]
        return predictions
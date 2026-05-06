def predict_condition_models(self, model_names,
                                 input_columns,
                                 metadata_cols,
                                 data_mode="forecast",
                                 ):
        """
        Apply condition modelsto forecast data.
        Args:
            model_names: List of names associated with each condition model used for prediction
            input_columns: List of columns in data used as input into the model
            metadata_cols: Columns from input data that should be included in the data frame with the predictions.
            data_mode: Which data subset to pull from for the predictions, "forecast" by default
        Returns:
            A dictionary of data frames containing probabilities of the event and specified metadata
        """
        groups = self.condition_models.keys()
        predictions = pd.DataFrame(self.data[data_mode]["combo"][metadata_cols])
        for group in groups:
            print(group)
            print(self.condition_models[group])
            g_idxs = self.data[data_mode]["combo"][self.group_col] == group
            group_count = np.count_nonzero(g_idxs)
            if group_count > 0:
                for m, model_name in enumerate(model_names):
                    mn = model_name.replace(" ", "-")
                    predictions.loc[g_idxs, mn + "_conditionprob"] = self.condition_models[group][
                                                                         model_name].predict_proba(
                        self.data[data_mode]["combo"].loc[g_idxs, input_columns])[:, 1]
                    predictions.loc[g_idxs,
                                    mn + "_conditionthresh"] = np.where(predictions.loc[g_idxs, mn + "_conditionprob"]
                                                                        >= self.condition_models[group][
                                                                            model_name + "_condition_threshold"], 1, 0)

        return predictions
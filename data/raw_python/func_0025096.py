def predict_size_models(self, model_names,
                            input_columns,
                            metadata_cols,
                            data_mode="forecast"):
        """
        Apply size models to forecast data.
        Args:
            model_names:
            input_columns:
            metadata_cols:
            data_mode:
        """
        groups = self.size_models.keys()
        predictions = {}
        for group in groups:
            group_data = self.data[data_mode]["combo"].loc[self.data[data_mode]["combo"][self.group_col] == group]
            if group_data.shape[0] > 0:
                predictions[group] = {}
                output_values = self.size_models[group]["outputvalues"].astype(int)
                for m, model_name in enumerate(model_names):
                    print("{0} {1}".format(group, model_name))
                    pred_col_names = [model_name.replace(" ", "-") + "_{0:02d}".format(p) for p in output_values]
                    predictions[group][model_name] = group_data[metadata_cols]
                    pred_vals = self.size_models[group][model_name].predict_proba(group_data[input_columns])
                    pred_classes = self.size_models[group][model_name].classes_
                    pred_pdf = np.zeros((pred_vals.shape[0], output_values.size))
                    for pcv, pc in enumerate(pred_classes):
                        idx = np.where(output_values == pc)[0][0]
                        pred_pdf[:, idx] = pred_vals[:, pcv]
                    for pcn, pred_col_name in enumerate(pred_col_names):
                        predictions[group][model_name].loc[:, pred_col_name] = pred_pdf[:, pcn]
        return predictions
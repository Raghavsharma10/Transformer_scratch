def merge_obs(self):
        """
        Match forecasts and observations.
        """
        for model_type in self.model_types:
            self.matched_forecasts[model_type] = {}
            for model_name in self.model_names[model_type]:
                self.matched_forecasts[model_type][model_name] = pd.merge(self.forecasts[model_type][model_name],
                                                                          self.obs, right_on="Step_ID", how="left",
                                                                          left_index=True)
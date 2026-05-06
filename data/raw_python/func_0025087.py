def calc_copulas(self,
                     output_file,
                     model_names=("start-time", "translation-x", "translation-y"),
                     label_columns=("Start_Time_Error", "Translation_Error_X", "Translation_Error_Y")):
        """
        Calculate a copula multivariate normal distribution from the training data for each group of ensemble members.
        Distributions are written to a pickle file for later use.
        Args:
            output_file: Pickle file
            model_names: Names of the tracking models
            label_columns: Names of the data columns used for labeling
        Returns:
        """
        if len(self.data['train']) == 0:
            self.load_data()
        groups = self.data["train"]["member"][self.group_col].unique()
        copulas = {}
        label_columns = list(label_columns)
        for group in groups:
            print(group)
            group_data = self.data["train"]["total_group"].loc[
                self.data["train"]["total_group"][self.group_col] == group]
            group_data = group_data.dropna()
            group_data.reset_index(drop=True, inplace=True)
            copulas[group] = {}
            copulas[group]["mean"] = group_data[label_columns].mean(axis=0).values
            copulas[group]["cov"] = np.cov(group_data[label_columns].values.T)
            copulas[group]["model_names"] = list(model_names)
            del group_data
        pickle.dump(copulas, open(output_file, "w"), pickle.HIGHEST_PROTOCOL)
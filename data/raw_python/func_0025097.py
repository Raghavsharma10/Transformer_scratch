def fit_track_models(self,
                         model_names,
                         model_objs,
                         input_columns,
                         output_columns,
                         output_ranges,
                         ):
        """
        Fit machine learning models to predict track error offsets.
            model_names:
            model_objs:
            input_columns:
            output_columns:
            output_ranges:
        """
        print("Fitting track models")
        groups = self.data["train"]["member"][self.group_col].unique()
        for group in groups:
            group_data = self.data["train"]["combo"].loc[self.data["train"]["combo"][self.group_col] == group]
            group_data = group_data.dropna()
            group_data = group_data.loc[group_data["Duration_Step"] == 1]
            for model_type, model_dict in self.track_models.items():
                model_dict[group] = {}
                output_data = group_data[output_columns[model_type]].values.astype(int)
                output_data[output_data < output_ranges[model_type][0]] = output_ranges[model_type][0]
                output_data[output_data > output_ranges[model_type][1]] = output_ranges[model_type][1]
                discrete_data = (output_data - output_ranges[model_type][0]) // output_ranges[model_type][2] * \
                                output_ranges[model_type][2] + output_ranges[model_type][0]
                model_dict[group]["outputvalues"] = np.arange(output_ranges[model_type][0],
                                                              output_ranges[model_type][1] +
                                                              output_ranges[model_type][2],
                                                              output_ranges[model_type][2])
                for m, model_name in enumerate(model_names):
                    print("{0} {1} {2}".format(group, model_type, model_name))
                    model_dict[group][model_name] = deepcopy(model_objs[m])
                    model_dict[group][model_name].fit(group_data[input_columns], discrete_data)
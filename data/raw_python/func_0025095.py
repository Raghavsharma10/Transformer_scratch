def fit_size_models(self, model_names,
                        model_objs,
                        input_columns,
                        output_column="Hail_Size",
                        output_start=5,
                        output_step=5,
                        output_stop=100):
        """
        Fit size models to produce discrete pdfs of forecast hail sizes.
        Args:
            model_names: List of model names
            model_objs: List of model objects
            input_columns: List of input variables
            output_column: Output variable name
            output_start: Hail size bin start
            output_step: hail size bin step
            output_stop: hail size bin stop
        """
        print("Fitting size models")
        groups = self.data["train"]["member"][self.group_col].unique()
        output_start = int(output_start)
        output_step = int(output_step)
        output_stop = int(output_stop)
        for group in groups:
            group_data = self.data["train"]["combo"].loc[self.data["train"]["combo"][self.group_col] == group]
            group_data.dropna(inplace=True)
            group_data = group_data[group_data[output_column] >= output_start]
            output_data = group_data[output_column].values.astype(int)
            output_data[output_data > output_stop] = output_stop
            discrete_data = ((output_data - output_start) // output_step) * output_step + output_start
            self.size_models[group] = {}
            self.size_models[group]["outputvalues"] = np.arange(output_start, output_stop + output_step, output_step,
                                                                dtype=int)
            for m, model_name in enumerate(model_names):
                print("{0} {1}".format(group, model_name))
                self.size_models[group][model_name] = deepcopy(model_objs[m])
                self.size_models[group][model_name].fit(group_data[input_columns], discrete_data)
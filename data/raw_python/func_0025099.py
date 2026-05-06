def load_models(self, model_path):
        """
        Load models from pickle files.
        """
        condition_model_files = sorted(glob(model_path + "*_condition.pkl"))
        if len(condition_model_files) > 0:
            for condition_model_file in condition_model_files:
                model_comps = condition_model_file.split("/")[-1][:-4].split("_")
                if model_comps[0] not in self.condition_models.keys():
                    self.condition_models[model_comps[0]] = {}
                model_name = model_comps[1].replace("-", " ")
                with open(condition_model_file, "rb") as cmf:
                    if "condition_threshold" in condition_model_file:
                        self.condition_models[model_comps[0]][model_name + "_condition_threshold"] = pickle.load(cmf)
                    else:
                        self.condition_models[model_comps[0]][model_name] = pickle.load(cmf)

        size_model_files = sorted(glob(model_path + "*_size.pkl"))
        if len(size_model_files) > 0:
            for size_model_file in size_model_files:
                model_comps = size_model_file.split("/")[-1][:-4].split("_")
                if model_comps[0] not in self.size_models.keys():
                    self.size_models[model_comps[0]] = {}
                model_name = model_comps[1].replace("-", " ")
                with open(size_model_file, "rb") as smf:
                    self.size_models[model_comps[0]][model_name] = pickle.load(smf)

        size_dist_model_files = sorted(glob(model_path + "*_sizedist.pkl"))
        if len(size_dist_model_files) > 0:
            for dist_model_file in size_dist_model_files:
                model_comps = dist_model_file.split("/")[-1][:-4].split("_")
                if model_comps[0] not in self.size_distribution_models.keys():
                    self.size_distribution_models[model_comps[0]] = {}
                if "_".join(model_comps[2:-1]) not in self.size_distribution_models[model_comps[0]].keys():
                    self.size_distribution_models[model_comps[0]]["_".join(model_comps[2:-1])] = {}
                model_name = model_comps[1].replace("-", " ")
                with open(dist_model_file, "rb") as dmf:
                    self.size_distribution_models[model_comps[0]]["_".join(model_comps[2:-1])][
                        model_name] = pickle.load(dmf)

        track_model_files = sorted(glob(model_path + "*_track.pkl"))
        if len(track_model_files) > 0:
            for track_model_file in track_model_files:
                model_comps = track_model_file.split("/")[-1][:-4].split("_")
                group = model_comps[0]
                model_name = model_comps[1].replace("-", " ")
                model_type = model_comps[2]
                if model_type not in self.track_models.keys():
                    self.track_models[model_type] = {}
                if group not in self.track_models[model_type].keys():
                    self.track_models[model_type][group] = {}
                with open(track_model_file, "rb") as tmf:
                    self.track_models[model_type][group][model_name] = pickle.load(tmf)
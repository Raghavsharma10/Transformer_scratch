def save_models(self, model_path):
        """
        Save machine learning models to pickle files.
        """
        for group, condition_model_set in self.condition_models.items():
            for model_name, model_obj in condition_model_set.items():
                out_filename = model_path + \
                               "{0}_{1}_condition.pkl".format(group,
                                                              model_name.replace(" ", "-"))
                with open(out_filename, "wb") as pickle_file:
                    pickle.dump(model_obj,
                                pickle_file,
                                pickle.HIGHEST_PROTOCOL)
        for group, size_model_set in self.size_models.items():
            for model_name, model_obj in size_model_set.items():
                out_filename = model_path + \
                               "{0}_{1}_size.pkl".format(group,
                                                         model_name.replace(" ", "-"))
                with open(out_filename, "wb") as pickle_file:
                    pickle.dump(model_obj,
                                pickle_file,
                                pickle.HIGHEST_PROTOCOL)
        for group, dist_model_set in self.size_distribution_models.items():
            for model_type, model_objs in dist_model_set.items():
                for model_name, model_obj in model_objs.items():
                    out_filename = model_path + \
                                   "{0}_{1}_{2}_sizedist.pkl".format(group,
                                                                     model_name.replace(" ", "-"),
                                                                     model_type)
                    with open(out_filename, "wb") as pickle_file:
                        pickle.dump(model_obj,
                                    pickle_file,
                                    pickle.HIGHEST_PROTOCOL)
        for model_type, track_type_models in self.track_models.items():
            for group, track_model_set in track_type_models.items():
                for model_name, model_obj in track_model_set.items():
                    out_filename = model_path + \
                                   "{0}_{1}_{2}_track.pkl".format(group,
                                                                  model_name.replace(" ", "-"),
                                                                  model_type)
                    with open(out_filename, "wb") as pickle_file:
                        pickle.dump(model_obj,
                                    pickle_file,
                                    pickle.HIGHEST_PROTOCOL)

        return
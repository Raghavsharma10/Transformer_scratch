def output_forecasts_json(self, forecasts,
                              condition_model_names,
                              size_model_names,
                              dist_model_names,
                              track_model_names,
                              json_data_path,
                              out_path):
        """
        Output forecast values to geoJSON file format.
        :param forecasts:
        :param condition_model_names:
        :param size_model_names:
        :param track_model_names:
        :param json_data_path:
        :param out_path:
        :return:
        """
        total_tracks = self.data["forecast"]["total"]
        for r in np.arange(total_tracks.shape[0]):
            track_id = total_tracks.loc[r, "Track_ID"]
            print(track_id)
            track_num = track_id.split("_")[-1]
            ensemble_name = total_tracks.loc[r, "Ensemble_Name"]
            member = total_tracks.loc[r, "Ensemble_Member"]
            group = self.data["forecast"]["member"].loc[self.data["forecast"]["member"]["Ensemble_Member"] == member,
                                                        self.group_col].values[0]
            run_date = track_id.split("_")[-4][:8]
            step_forecasts = {}
            for ml_model in condition_model_names:
                step_forecasts["condition_" + ml_model.replace(" ", "-")] = forecasts["condition"][group].loc[
                    forecasts["condition"][group]["Track_ID"] == track_id, ml_model]
            for ml_model in size_model_names:
                step_forecasts["size_" + ml_model.replace(" ", "-")] = forecasts["size"][group][ml_model].loc[
                    forecasts["size"][group][ml_model]["Track_ID"] == track_id]
            for ml_model in dist_model_names:
                step_forecasts["dist_" + ml_model.replace(" ", "-")] = forecasts["dist"][group][ml_model].loc[
                    forecasts["dist"][group][ml_model]["Track_ID"] == track_id]
            for model_type in forecasts["track"].keys():
                for ml_model in track_model_names:
                    mframe = forecasts["track"][model_type][group][ml_model]
                    step_forecasts[model_type + "_" + ml_model.replace(" ", "-")] = mframe.loc[
                        mframe["Track_ID"] == track_id]
            json_file_name = "{0}_{1}_{2}_model_track_{3}.json".format(ensemble_name,
                                                                       run_date,
                                                                       member,
                                                                       track_num)
            full_json_path = json_data_path + "/".join([run_date, member]) + "/" + json_file_name
            with open(full_json_path) as json_file_obj:
                try:
                    track_obj = json.load(json_file_obj)
                except FileNotFoundError:
                    print(full_json_path + " not found")
                    continue
            for f, feature in enumerate(track_obj['features']):
                del feature['properties']['attributes']
                for model_name, fdata in step_forecasts.items():
                    ml_model_name = model_name.split("_")[1]
                    if "condition" in model_name:
                        feature['properties'][model_name] = fdata.values[f]
                    else:
                        predcols = []
                        for col in fdata.columns:
                            if ml_model_name in col:
                                predcols.append(col)
                        feature['properties'][model_name] = fdata.loc[:, predcols].values[f].tolist()
            full_path = []
            for part in [run_date, member]:
                full_path.append(part)
                if not os.access(out_path + "/".join(full_path), os.R_OK):
                    try:
                        os.mkdir(out_path + "/".join(full_path))
                    except OSError:
                        print("directory already created")
            out_json_filename = out_path + "/".join(full_path) + "/" + json_file_name
            with open(out_json_filename, "w") as out_json_obj:
                json.dump(track_obj, out_json_obj, indent=1, sort_keys=True)
        return
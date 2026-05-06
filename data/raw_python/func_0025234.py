def merge_input_csv_forecast_json(input_csv_file, forecast_json_path, condition_models, dist_models):
    """
    Reads forecasts from json files and merges them with the input data from the step csv files.

    Args:
        input_csv_file: Name of the input data csv file being processed
        forecast_json_path: Path to the forecast json files toplevel directory
        condition_models: List of models used to forecast hail or no hail
        dist_models: List of models used to forecast the hail size distribution

    Returns:

    """
    try:
        run_date = input_csv_file[:-4].split("_")[-1]
        print(run_date)
        ens_member = "_".join(input_csv_file.split("/")[-1][:-4].split("_")[3:-1])
        ens_name = input_csv_file.split("/")[-1].split("_")[2]
        input_data = pd.read_csv(input_csv_file, index_col="Step_ID")
        full_json_path = forecast_json_path + "{0}/{1}/".format(run_date, ens_member)
        track_ids = sorted(input_data["Track_ID"].unique())
        model_pred_cols = []
        condition_models_ns = []
        dist_models_ns = []
        gamma_params = ["Shape", "Location", "Scale"]
        for condition_model in condition_models:
            model_pred_cols.append(condition_model.replace(" ", "-") + "_Condition")
            condition_models_ns.append(condition_model.replace(" ", "-"))
        for dist_model in dist_models:
            dist_models_ns.append(dist_model.replace(" ", "-"))
            for param in gamma_params:
                model_pred_cols.append(dist_model.replace(" ", "-") + "_" + param)
        pred_data = pd.DataFrame(index=input_data.index, columns=model_pred_cols,
                                dtype=float)
        for track_id in track_ids:
            track_id_num = track_id.split("_")[-1]
            json_filename = full_json_path + "{0}_{1}_{2}_model_track_{3}.json".format(ens_name,
                                                                                    run_date,
                                                                                    ens_member,
                                                                                    track_id_num)
            json_file = open(json_filename)
            json_data = json.load(json_file)
            json_file.close()
            for s, step in enumerate(json_data["features"]):
                step_id = track_id + "_{0:02d}".format(s)
                for cond_model in condition_models_ns:
                    pred_data.loc[step_id, cond_model + "_Condition"]  = step["properties"]["condition_" + cond_model]
                for dist_model in dist_models_ns:
                    pred_data.loc[step_id, [dist_model + "_" + p
                                            for p in gamma_params]] = step["properties"]["dist_" + dist_model]
        out_data = input_data.merge(pred_data, left_index=True, right_index=True)
        return out_data, ens_name, ens_member
    except Exception as e:
        print(traceback.format_exc())
        raise e
def load_obs(self):
        """
        Loads the track total and step files and merges the information into a single data frame.
        """
        track_total_file = self.track_data_csv_path + \
            "track_total_{0}_{1}_{2}.csv".format(self.ensemble_name,
                                                 self.ensemble_member,
                                                 self.run_date.strftime("%Y%m%d"))
        track_step_file = self.track_data_csv_path + \
            "track_step_{0}_{1}_{2}.csv".format(self.ensemble_name,
                                                self.ensemble_member,
                                                self.run_date.strftime("%Y%m%d"))
        track_total_cols = ["Track_ID", "Translation_Error_X", "Translation_Error_Y", "Start_Time_Error"]
        track_step_cols = ["Step_ID", "Track_ID", "Hail_Size", "Shape", "Location", "Scale"]
        track_total_data = pd.read_csv(track_total_file, usecols=track_total_cols)
        track_step_data = pd.read_csv(track_step_file, usecols=track_step_cols)
        obs_data = pd.merge(track_step_data, track_total_data, on="Track_ID", how="left")
        self.obs = obs_data
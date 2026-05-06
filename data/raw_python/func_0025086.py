def load_data(self, mode="train", format="csv"):
        """
        Load data from flat data files containing total track information and information about each timestep.
        The two sets are combined using merge operations on the Track IDs. Additional member information is gathered
        from the appropriate member file.
        Args:
            mode: "train" or "forecast"
            format:  file format being used. Default is "csv"
        """
        if mode in self.data.keys():
            run_dates = pd.DatetimeIndex(start=self.start_dates[mode],
                                        end=self.end_dates[mode],freq="1D")
            run_date_str = [d.strftime("%Y%m%d-%H%M") for d in run_dates.date]
            print(run_date_str)
            all_total_track_files = sorted(glob(getattr(self, mode + "_data_path") +
                                                "*total_" + self.ensemble_name + "*." + format))
            all_step_track_files = sorted(glob(getattr(self, mode + "_data_path") +
                                               "*step_" + self.ensemble_name + "*." + format))
            total_track_files = []
            for track_file in all_total_track_files:
                file_date = track_file.split("_")[-1][:-4]
                if file_date in run_date_str:
                    total_track_files.append(track_file)
            step_track_files = []
            for step_file in all_step_track_files:
                file_date = step_file.split("_")[-1][:-4]
                if file_date in run_date_str:
                    step_track_files.append(step_file)            
	  	   
            self.data[mode]["total"] = pd.concat(map(pd.read_csv, total_track_files),
                                                 ignore_index=True)
            self.data[mode]["total"] = self.data[mode]["total"].fillna(value=0)
            self.data[mode]["total"] = self.data[mode]["total"].replace([np.inf, -np.inf], 0)
            self.data[mode]["step"] = pd.concat(map(pd.read_csv, step_track_files),
                                                ignore_index=True)
            self.data[mode]["step"] = self.data[mode]["step"].fillna(value=0)
            self.data[mode]["step"] = self.data[mode]["step"].replace([np.inf, -np.inf], 0)
            if mode == "forecast":
                self.data[mode]["step"] = self.data[mode]["step"].drop_duplicates("Step_ID")
            self.data[mode]["member"] = pd.read_csv(self.member_files[mode])
            self.data[mode]["combo"] = pd.merge(self.data[mode]["step"],
                                                self.data[mode]["total"],
                                                on=["Track_ID", "Ensemble_Name", "Ensemble_Member", "Run_Date"])
            self.data[mode]["combo"] = pd.merge(self.data[mode]["combo"],
                                                self.data[mode]["member"],
                                                on="Ensemble_Member") 
            self.data[mode]["total_group"] = pd.merge(self.data[mode]["total"],
                                                      self.data[mode]["member"],
                                                      on="Ensemble_Member")
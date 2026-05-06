def output_forecasts_csv(self, forecasts, mode, csv_path, run_date_format="%Y%m%d-%H%M"):
        """
        Output hail forecast values to csv files by run date and ensemble member.

        Args:
            forecasts:
            mode:
            csv_path:
        Returns:
        """
        merged_forecasts = pd.merge(forecasts["condition"],
                                    forecasts["dist"],
                                    on=["Step_ID","Track_ID","Ensemble_Member","Forecast_Hour"])
        all_members = self.data[mode]["combo"]["Ensemble_Member"]
        members = np.unique(all_members)
        all_run_dates = pd.DatetimeIndex(self.data[mode]["combo"]["Run_Date"])
        run_dates = pd.DatetimeIndex(np.unique(all_run_dates))
        print(run_dates)
        for member in members:
            for run_date in run_dates:
                mem_run_index = (all_run_dates == run_date) & (all_members == member)
                member_forecast = merged_forecasts.loc[mem_run_index]
                member_forecast.to_csv(join(csv_path, "hail_forecasts_{0}_{1}_{2}.csv".format(self.ensemble_name,
                                                                                              member,
                                                                                              run_date.strftime
																							  (run_date_format))))
        return
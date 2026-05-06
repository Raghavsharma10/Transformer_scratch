def load_obs(self,  mask_threshold=0.5):
        """
        Loads observations and masking grid (if needed).

        :param mask_threshold: Values greater than the threshold are kept, others are masked.
        :return:
        """
        start_date = self.run_date + timedelta(hours=self.start_hour)
        end_date = self.run_date + timedelta(hours=self.end_hour)
        mrms_grid = MRMSGrid(start_date, end_date, self.mrms_variable, self.mrms_path)
        mrms_grid.load_data()
        if len(mrms_grid.data) > 0:
            self.raw_obs[self.mrms_variable] = np.where(mrms_grid.data > 100, 100, mrms_grid.data)
            self.window_obs[self.mrms_variable] = np.array([self.raw_obs[self.mrms_variable][sl].max(axis=0)
                                                            for sl in self.hour_windows])
            if self.obs_mask:
                mask_grid = MRMSGrid(start_date, end_date, self.mask_variable, self.mrms_path)
                mask_grid.load_data()
                self.raw_obs[self.mask_variable] = np.where(mask_grid.data >= mask_threshold, 1, 0)
                self.window_obs[self.mask_variable] = np.array([self.raw_obs[self.mask_variable][sl].max(axis=0)
                                                               for sl in self.hour_windows])
def find_model_patch_tracks(self):
        """
        Identify storms in gridded model output and extract uniform sized patches around the storm centers of mass.

        Returns:

        """
        self.model_grid.load_data()
        tracked_model_objects = []
        model_objects = []
        if self.model_grid.data is None:
            print("No model output found")
            return tracked_model_objects
        min_orig = self.model_ew.min_thresh
        max_orig = self.model_ew.max_thresh
        data_increment_orig = self.model_ew.data_increment
        self.model_ew.min_thresh = 0
        self.model_ew.data_increment = 1
        self.model_ew.max_thresh = 100
        for h, hour in enumerate(self.hours):
            # Identify storms at each time step and apply size filter
            print("Finding {0} objects for run {1} Hour: {2:02d}".format(self.ensemble_member,
                                                                         self.run_date.strftime("%Y%m%d%H"), hour))
            if self.mask is not None:
                model_data = self.model_grid.data[h] * self.mask
            else:
                model_data = self.model_grid.data[h]
            model_data[:self.patch_radius] = 0
            model_data[-self.patch_radius:] = 0
            model_data[:, :self.patch_radius] = 0
            model_data[:, -self.patch_radius:] = 0
            scaled_data = np.array(rescale_data(model_data, min_orig, max_orig))
            hour_labels = label_storm_objects(scaled_data, "ew",
                                              self.model_ew.min_thresh, self.model_ew.max_thresh,
                                              min_area=self.size_filter, max_area=self.model_ew.max_size,
                                              max_range=self.model_ew.delta, increment=self.model_ew.data_increment,
                                              gaussian_sd=self.gaussian_window)
            model_objects.extend(extract_storm_patches(hour_labels, model_data, self.model_grid.x,
                                                       self.model_grid.y, [hour],
                                                       dx=self.model_grid.dx,
                                                       patch_radius=self.patch_radius))
            for model_obj in model_objects[-1]:
                dims = model_obj.timesteps[-1].shape
                if h > 0:
                    model_obj.estimate_motion(hour, self.model_grid.data[h-1], dims[1], dims[0])
            del scaled_data
            del model_data
            del hour_labels
        tracked_model_objects.extend(track_storms(model_objects, self.hours,
                                                  self.object_matcher.cost_function_components,
                                                  self.object_matcher.max_values,
                                                  self.object_matcher.weights))
        self.model_ew.min_thresh = min_orig
        self.model_ew.max_thresh = max_orig
        self.model_ew.data_increment = data_increment_orig
        return tracked_model_objects
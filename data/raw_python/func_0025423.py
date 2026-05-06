def find_model_tracks(self):
        """
        Identify storms at each model time step and link them together with object matching.

        Returns:
            List of STObjects containing model track information.
        """
        self.model_grid.load_data()
        model_objects = []
        tracked_model_objects = []
        if self.model_grid.data is None:
            print("No model output found")
            return tracked_model_objects
        for h, hour in enumerate(self.hours):
            # Identify storms at each time step and apply size filter
            print("Finding {0} objects for run {1} Hour: {2:02d}".format(self.ensemble_member,
                                                                         self.run_date.strftime("%Y%m%d%H"), hour))
            if self.mask is not None:
                model_data = self.model_grid.data[h] * self.mask
            else:
                model_data = self.model_grid.data[h]

            # remember orig values
            min_orig = self.model_ew.min_thresh
            max_orig = self.model_ew.max_thresh
            data_increment_orig = self.model_ew.data_increment
            # scale to int 0-100.
            scaled_data = np.array(rescale_data( self.model_grid.data[h], min_orig, max_orig))
            self.model_ew.min_thresh = 0
            self.model_ew.data_increment = 1
            self.model_ew.max_thresh = 100
            hour_labels = self.model_ew.label(gaussian_filter(scaled_data, self.gaussian_window))
            hour_labels[model_data < self.model_ew.min_thresh] = 0
            hour_labels = self.model_ew.size_filter(hour_labels, self.size_filter)
            # Return to orig values
            self.model_ew.min_thresh = min_orig
            self.model_ew.max_thresh = max_orig
            self.model_ew.data_increment = data_increment_orig
            obj_slices = find_objects(hour_labels)

            num_slices = len(obj_slices)
            model_objects.append([])
            if num_slices > 0:
                for s, sl in enumerate(obj_slices):
                    model_objects[-1].append(STObject(self.model_grid.data[h][sl],
                                                      np.where(hour_labels[sl] == s + 1, 1, 0),
                                                      self.model_grid.x[sl], 
                                                      self.model_grid.y[sl], 
                                                      self.model_grid.i[sl], 
                                                      self.model_grid.j[sl],
                                                      hour,
                                                      hour,
                                                      dx=self.model_grid.dx))
                    if h > 0:
                        dims = model_objects[-1][-1].timesteps[0].shape
                        model_objects[-1][-1].estimate_motion(hour, self.model_grid.data[h-1], dims[1], dims[0])
            del hour_labels
            del scaled_data
            del model_data
        for h, hour in enumerate(self.hours):
            past_time_objs = []
            for obj in tracked_model_objects:
                # Potential trackable objects are identified
                if obj.end_time == hour - 1:
                    past_time_objs.append(obj)
            # If no objects existed in the last time step, then consider objects in current time step all new
            if len(past_time_objs) == 0:
                tracked_model_objects.extend(model_objects[h])
            # Match from previous time step with current time step
            elif len(past_time_objs) > 0 and len(model_objects[h]) > 0:
                assignments = self.object_matcher.match_objects(past_time_objs, model_objects[h], hour - 1, hour)
                unpaired = list(range(len(model_objects[h])))
                for pair in assignments:
                    past_time_objs[pair[0]].extend(model_objects[h][pair[1]])
                    unpaired.remove(pair[1])
                if len(unpaired) > 0:
                    for up in unpaired:
                        tracked_model_objects.append(model_objects[h][up])
            print("Tracked Model Objects: {0:03d} Hour: {1:02d}".format(len(tracked_model_objects), hour))

        return tracked_model_objects
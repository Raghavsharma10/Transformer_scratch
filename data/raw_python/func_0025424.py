def find_mrms_tracks(self):
        """
        Identify objects from MRMS timesteps and link them together with object matching.

        Returns:
            List of STObjects containing MESH track information.
        """
        obs_objects = []
        tracked_obs_objects = []
        if self.mrms_ew is not None:
            self.mrms_grid.load_data()
            
            if len(self.mrms_grid.data) != len(self.hours):
                print('Less than 24 hours of observation data found')
                
                return tracked_obs_objects
         
            for h, hour in enumerate(self.hours):
                mrms_data = np.zeros(self.mrms_grid.data[h].shape)
                mrms_data[:] = np.array(self.mrms_grid.data[h])
                mrms_data[mrms_data < 0] = 0
                hour_labels = self.mrms_ew.size_filter(self.mrms_ew.label(gaussian_filter(mrms_data,
                                                                                      self.gaussian_window)),
                                                       self.size_filter)
                hour_labels[mrms_data < self.mrms_ew.min_thresh] = 0
                obj_slices = find_objects(hour_labels)
                num_slices = len(obj_slices)
                obs_objects.append([])
                if num_slices > 0:
                    for sl in obj_slices:
                        obs_objects[-1].append(STObject(mrms_data[sl],
                                                        np.where(hour_labels[sl] > 0, 1, 0),
                                                        self.model_grid.x[sl],
                                                        self.model_grid.y[sl],
                                                        self.model_grid.i[sl],
                                                        self.model_grid.j[sl],
                                                        hour,
                                                        hour,
                                                        dx=self.model_grid.dx))
                        if h > 0:
                            dims = obs_objects[-1][-1].timesteps[0].shape
                            obs_objects[-1][-1].estimate_motion(hour, self.mrms_grid.data[h-1], dims[1], dims[0])
        
            for h, hour in enumerate(self.hours):
                past_time_objs = []
                for obj in tracked_obs_objects:
                    if obj.end_time == hour - 1:
                        past_time_objs.append(obj)
                if len(past_time_objs) == 0:
                    tracked_obs_objects.extend(obs_objects[h])
                elif len(past_time_objs) > 0 and len(obs_objects[h]) > 0:
                    assignments = self.object_matcher.match_objects(past_time_objs, obs_objects[h], hour - 1, hour)
                    unpaired = list(range(len(obs_objects[h])))
                    for pair in assignments:
                        past_time_objs[pair[0]].extend(obs_objects[h][pair[1]])
                        unpaired.remove(pair[1])
                    if len(unpaired) > 0:
                        for up in unpaired:
                            tracked_obs_objects.append(obs_objects[h][up])
                print("Tracked Obs Objects: {0:03d} Hour: {1:02d}".format(len(tracked_obs_objects), hour))
        
        return tracked_obs_objects
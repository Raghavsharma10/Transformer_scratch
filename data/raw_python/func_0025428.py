def match_hail_size_step_distributions(self, model_tracks, obs_tracks, track_pairings):
        """
        Given a matching set of observed tracks for each model track, 
        
        Args:
            model_tracks: 
            obs_tracks: 
            track_pairings: 

        Returns:

        """
        label_columns = ["Matched", "Max_Hail_Size", "Num_Matches", "Shape", "Location", "Scale"]
        s = 0
        for m, model_track in enumerate(model_tracks):
            model_track.observations = pd.DataFrame(index=model_track.times, columns=label_columns, dtype=np.float64)
            model_track.observations.loc[:, :] = 0
            model_track.observations["Matched"] = model_track.observations["Matched"].astype(np.int32)
            for t, time in enumerate(model_track.times):
                model_track.observations.loc[time, "Matched"] = track_pairings.loc[s, "Matched"]
                if model_track.observations.loc[time, "Matched"] > 0:
                    all_hail_sizes = []
                    step_pairs = track_pairings.loc[s, "Pairings"]
                    for step_pair in step_pairs:
                        obs_step = obs_tracks[step_pair[0]].timesteps[step_pair[1]].ravel()
                        obs_mask = obs_tracks[step_pair[0]].masks[step_pair[1]].ravel()
                        all_hail_sizes.append(obs_step[(obs_mask == 1) & (obs_step >= self.mrms_ew.min_thresh)])
                    combined_hail_sizes = np.concatenate(all_hail_sizes)
                    min_hail = combined_hail_sizes.min() - 0.1
                    model_track.observations.loc[time, "Max_Hail_Size"] = combined_hail_sizes.max()
                    model_track.observations.loc[time, "Num_Matches"] = step_pairs.shape[0]
                    model_track.observations.loc[time, ["Shape", "Location", "Scale"]] = gamma.fit(combined_hail_sizes,
                                                                                                   floc=min_hail)
                s += 1
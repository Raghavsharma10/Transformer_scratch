def match_hail_sizes(model_tracks, obs_tracks, track_pairings):
        """
        Given forecast and observed track pairings, maximum hail sizes are associated with each paired forecast storm
        track timestep. If the duration of the forecast and observed tracks differ, then interpolation is used for the
        intermediate timesteps.

        Args:
            model_tracks: List of model track STObjects
            obs_tracks: List of observed STObjects
            track_pairings: list of tuples containing the indices of the paired (forecast, observed) tracks
        """
        unpaired = list(range(len(model_tracks)))
        for p, pair in enumerate(track_pairings):
            model_track = model_tracks[pair[0]]
            unpaired.remove(pair[0])
            obs_track = obs_tracks[pair[1]]
            obs_hail_sizes = np.array([step[obs_track.masks[t] == 1].max()
                                       for t, step in enumerate(obs_track.timesteps)])
            if obs_track.times.size > 1 and model_track.times.size > 1:
                normalized_obs_times = 1.0 / (obs_track.times.max() - obs_track.times.min())\
                    * (obs_track.times - obs_track.times.min())
                normalized_model_times = 1.0 / (model_track.times.max() - model_track.times.min())\
                    * (model_track.times - model_track.times.min())
                hail_interp = interp1d(normalized_obs_times, obs_hail_sizes, kind="nearest",
                                       bounds_error=False, fill_value=0)
                model_track.observations = hail_interp(normalized_model_times)
            elif obs_track.times.size == 1:
                model_track.observations = np.ones(model_track.times.shape) * obs_hail_sizes[0]
            elif model_track.times.size == 1:
                model_track.observations = np.array([obs_hail_sizes.max()])
            print(pair[0], "obs",  obs_hail_sizes)
            print(pair[0], "model", model_track.observations)
        for u in unpaired:
            model_tracks[u].observations = np.zeros(model_tracks[u].times.shape)
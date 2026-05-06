def match(self, set_a, set_b):
        """
        For each step in each track from set_a, identify all steps in all tracks from set_b that meet all
        cost function criteria
        
        Args:
            set_a: List of STObjects
            set_b: List of STObjects

        Returns:
            track_pairings: pandas.DataFrame 
        """
        track_step_matches = [[] * len(set_a)]

        costs = self.cost_matrix(set_a, set_b)
        valid_costs = np.all(costs < 1, axis=2)
        set_a_matches, set_b_matches = np.where(valid_costs)
        s = 0
        track_pairings = pd.DataFrame(index=np.arange(costs.shape[0]),
                                      columns=["Track", "Step", "Time", "Matched", "Pairings"], dtype=object)
        set_b_info = []
        for trb, track_b in enumerate(set_b):
            for t, time in enumerate(track_b.times):
                set_b_info.append((trb, t))
        set_b_info_arr = np.array(set_b_info, dtype=int)
        for tr, track_a in enumerate(set_a):
            for t, time in enumerate(track_a.times):
                track_pairings.loc[s, ["Track", "Step", "Time"]] = [tr, t, time]
                track_pairings.loc[s, "Matched"] = 1 if np.count_nonzero(set_a_matches == s) > 0 else 0
                if track_pairings.loc[s, "Matched"] == 1:
                    track_pairings.loc[s, "Pairings"] = set_b_info_arr[set_b_matches[set_a_matches == s]]
                else:
                    track_pairings.loc[s, "Pairings"] = np.array([])
                s += 1
        return track_pairings
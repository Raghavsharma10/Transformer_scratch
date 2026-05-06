def match_tracks(self, model_tracks, obs_tracks, unique_matches=True, closest_matches=False):
        """
        Match forecast and observed tracks.

        Args:
            model_tracks:
            obs_tracks:
            unique_matches:
            closest_matches:

        Returns:

        """
        if unique_matches:
            pairings = self.track_matcher.match_tracks(model_tracks, obs_tracks, closest_matches=closest_matches)
        else:
            pairings = self.track_matcher.neighbor_matches(model_tracks, obs_tracks)
        return pairings
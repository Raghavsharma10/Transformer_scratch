def best_trial_tid(self, rank=0):
        """Get tid of the best trial

        rank=0 means the best model
        rank=1 means second best
        ...
        """
        candidates = [t for t in self.trials
                      if t['result']['status'] == STATUS_OK]
        if len(candidates) == 0:
            return None
        losses = [float(t['result']['loss']) for t in candidates]
        assert not np.any(np.isnan(losses))
        lid = np.where(np.argsort(losses).argsort() == rank)[0][0]
        return candidates[lid]["tid"]
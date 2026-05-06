def n_ok(self):
        """Number of ok trials()
        """
        if len(self.trials) == 0:
            return 0
        else:
            return np.sum(np.array(self.statuses()) == "ok")
def load_model(self, tid, custom_objects=None):
        """Load saved keras model of the trial.

        If tid = None, get the best model

        Not applicable for trials ran in cross validion (i.e. not applicable
        for `CompileFN.cv_n_folds is None`
        """
        if tid is None:
            tid = self.best_trial_tid()

        model_path = self.get_trial(tid)["result"]["path"]["model"]
        return load_model(model_path, custom_objects=custom_objects)
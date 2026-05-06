def roc_curves(self, prob_thresholds):
        """
        Generate ROC Curve objects for each machine learning model, size threshold, and time window.

        :param prob_thresholds: Probability thresholds for the ROC Curve
        :param dilation_radius: Number of times to dilate the observation grid.
        :return: a dictionary of DistributedROC objects.
        """
        all_roc_curves = {}
        for model_name in self.model_names:
            all_roc_curves[model_name] = {}
            for size_threshold in self.size_thresholds:
                all_roc_curves[model_name][size_threshold] = {}
                for h, hour_window in enumerate(self.hour_windows):
                    hour_range = (hour_window.start, hour_window.stop)
                    all_roc_curves[model_name][size_threshold][hour_range] = \
                        DistributedROC(prob_thresholds, 1)
                    if self.obs_mask:
                        all_roc_curves[model_name][size_threshold][hour_range].update(
                            self.window_forecasts[model_name][size_threshold][h][
                                self.window_obs[self.mask_variable][h] > 0],
                            self.dilated_obs[size_threshold][h][self.window_obs[self.mask_variable][h] > 0]
                        )
                    else:
                        all_roc_curves[model_name][size_threshold][hour_range].update(
                            self.window_forecasts[model_name][size_threshold][h],
                            self.dilated_obs[size_threshold][h]
                        )
        return all_roc_curves
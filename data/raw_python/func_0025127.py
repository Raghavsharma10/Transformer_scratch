def reliability_curves(self, prob_thresholds):
        """
        Output reliability curves for each machine learning model, size threshold, and time window.

        :param prob_thresholds:
        :param dilation_radius:
        :return:
        """
        all_rel_curves = {}
        for model_name in self.model_names:
            all_rel_curves[model_name] = {}
            for size_threshold in self.size_thresholds:
                all_rel_curves[model_name][size_threshold] = {}
                for h, hour_window in enumerate(self.hour_windows):
                    hour_range = (hour_window.start, hour_window.stop)
                    all_rel_curves[model_name][size_threshold][hour_range] = \
                        DistributedReliability(prob_thresholds, 1)
                    if self.obs_mask:
                        all_rel_curves[model_name][size_threshold][hour_range].update(
                            self.window_forecasts[model_name][size_threshold][h][
                                self.window_obs[self.mask_variable][h] > 0],
                            self.dilated_obs[size_threshold][h][self.window_obs[self.mask_variable][h] > 0]
                        )
                    else:
                        all_rel_curves[model_name][size_threshold][hour_range].update(
                            self.window_forecasts[model_name][size_threshold][h],
                            self.dilated_obs[size_threshold][h]
                        )
        return all_rel_curves
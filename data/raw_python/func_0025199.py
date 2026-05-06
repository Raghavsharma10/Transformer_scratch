def calc_shape_step(self, stat_names, time):
        """
        Calculate shape statistics for a single time step

        Args:
            stat_names: List of shape statistics calculated from region props
            time: Time being investigated

        Returns:
            List of shape statistics

        """
        ti = np.where(self.times == time)[0][0]
        props = regionprops(self.masks[ti], self.timesteps[ti])[0]
        shape_stats = []
        for stat_name in stat_names:
            if "moments_hu" in stat_name:
                hu_index = int(stat_name.split("_")[-1])
                hu_name = "_".join(stat_name.split("_")[:-1])
                hu_val = np.log(props[hu_name][hu_index])
                if np.isnan(hu_val):
                    shape_stats.append(0)
                else:
                    shape_stats.append(hu_val)
            else:
                shape_stats.append(props[stat_name])
        return shape_stats
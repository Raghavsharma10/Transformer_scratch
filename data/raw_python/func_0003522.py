def land_threshold(self, land_cloud_prob, pcps, water):
        """Dynamic threshold for determining cloud cutoff
        Equation 17 (Zhu and Woodcock, 2012)
        Parameters
        ----------
        land_cloud_prob: ndarray
            probability of cloud over land
        pcps: ndarray
            potential cloud pixels
        water: ndarray
            water mask
        Output
        ------
        float:
            land cloud threshold
        """
        # eq 12
        clearsky_land = ~(pcps | water)

        # 82.5th percentile of lCloud_Prob(masked by clearsky_land) + LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF.2
        cloud_prob = land_cloud_prob.copy()
        cloud_prob[~clearsky_land] = np.nan
        cloud_prob[~self.mask] = np.nan

        # eq 17
        th_const = 0.2
        return np.nanpercentile(cloud_prob, 82.5) + th_const
def potential_cloud_layer(self, pcp, water, tlow, land_cloud_prob, land_threshold,
            water_cloud_prob, water_threshold=0.5):
        """Final step of determining potential cloud layer
        Equation 18 (Zhu and Woodcock, 2012)
        
        Saturation (green or red) test is not in the algorithm
        
        Parameters
        ----------
        pcps: ndarray
            potential cloud pixels
        water: ndarray
            water mask
        tirs1: ndarray
        tlow: float
            low percentile of land temperature
        land_cloud_prob: ndarray
            probability of cloud over land
        land_threshold: float
            cutoff for cloud over land
        water_cloud_prob: ndarray
            probability of cloud over water
        water_threshold: float
            cutoff for cloud over water
        Output
        ------
        ndarray:
            potential cloud layer, boolean
        """
        # Using pcp and water as mask todo
        # change water threshold to dynamic, line 132 in Zhu, 2015 todo
        part1 = (pcp & water & (water_cloud_prob > water_threshold))
        part2 = (pcp & ~water & (land_cloud_prob > land_threshold))
        temptest = self.tirs1 < (tlow - 35)  # 35degrees C colder

        if self.sat in ['LT5', 'LE7']:
            saturation = self.blue_saturated | self.green_saturated | self.red_saturated

            return part1 | part2 | temptest | saturation

        else:
            return part1 | part2 | temptest
def cloud_mask(self, min_filter=(3, 3), max_filter=(10, 10), combined=False, cloud_and_shadow=False):
        """Calculate the potential cloud layer from source data
        *This is the high level function which ties together all
        the equations for generating potential clouds*
        Parameters
        ----------
        blue: ndarray
        green: ndarray
        red: ndarray
        nir: ndarray
        swir1: ndarray
        swir2: ndarray
        cirrus: ndarray
        tirs1: ndarray
        min_filter: 2-element tuple, default=(3,3)
            Defines the window for the minimum_filter, for removing outliers
        max_filter: 2-element tuple, default=(21, 21)
            Defines the window for the maximum_filter, for "buffering" the edges
        combined: make a boolean array masking all (cloud, shadow, water)
        Output
        ------
        ndarray, boolean:
            potential cloud layer; True = cloud
        ndarray, boolean
            potential cloud shadow layer; True = cloud shadow
            :param cloud_and_shadow:
        """
        # logger.info("Running initial testsr")
        whiteness = self.whiteness_index()
        water = self.water_test()

        # First pass, potential clouds
        pcps = self.potential_cloud_pixels()

        if self.sat == 'LC8':
            cirrus_prob = self.cirrus / 0.04
        else:
            cirrus_prob = 0.0

        # Clouds over water
        wtp = self.water_temp_prob()
        bp = self.brightness_prob()
        water_cloud_prob = (wtp * bp) + cirrus_prob
        wthreshold = 0.5

        # Clouds over land
        tlow, thigh = self.temp_land(pcps, water)
        ltp = self.land_temp_prob(tlow, thigh)
        vp = self.variability_prob(whiteness)
        land_cloud_prob = (ltp * vp) + cirrus_prob
        lthreshold = self.land_threshold(land_cloud_prob, pcps, water)

        # logger.info("Calculate potential clouds")
        pcloud = self.potential_cloud_layer(
            pcps, water, tlow,
            land_cloud_prob, lthreshold,
            water_cloud_prob, wthreshold)

        # Ignoring snow for now as it exhibits many false positives and negatives
        # when used as a binary mask
        # psnow = potential_snow_layer(ndsi, green, nir, tirs1)
        # pcloud = pcloud & ~psnow

        # logger.info("Calculate potential cloud shadows")
        pshadow = self.potential_cloud_shadow_layer(water)

        # The remainder of the algorithm differs significantly from Fmask
        # In an attempt to make a more visually appealling cloud mask
        # with fewer inclusions and more broad shapes

        if min_filter:
            # Remove outliers
            # logger.info("Remove outliers with minimum filter")

            from scipy.ndimage.filters import minimum_filter
            from scipy.ndimage.morphology import distance_transform_edt

            # remove cloud outliers by nibbling the edges
            pcloud = minimum_filter(pcloud, size=min_filter)

            # crude, just look x pixels away for potential cloud pixels
            dist = distance_transform_edt(~pcloud)
            pixel_radius = 100.0
            pshadow = (dist < pixel_radius) & pshadow

            # remove cloud shadow outliers
            pshadow = minimum_filter(pshadow, size=min_filter)

        if max_filter:
            # grow around the edges
            # logger.info("Buffer edges with maximum filter")

            from scipy.ndimage.filters import maximum_filter

            pcloud = maximum_filter(pcloud, size=max_filter)
            pshadow = maximum_filter(pshadow, size=max_filter)

        # mystery, save pcloud here, shows no nan in qgis, save later, shows nan
        # outfile = '/data01/images/sandbox/pcloud.tif'
        # georeference = self.sat_image.rasterio_geometry
        # array = pcloud
        # array = array.reshape(1, array.shape[LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF], array.shape[1])
        # array = np.array(array, dtype=georeference['dtype'])
        # with rasterio.open(outfile, 'w', **georeference) as dst:
        #     dst.write(array)
        # mystery test
        if combined:
            return pcloud | pshadow | water

        if cloud_and_shadow:
            return pcloud | pshadow

        return pcloud, pshadow, water
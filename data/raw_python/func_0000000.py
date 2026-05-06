def __msgc_step3_discontinuity_localization(self):
        """
        Estimate discontinuity in basis of low resolution image segmentation.
        :return: discontinuity in low resolution
        """
        import scipy

        start = self._start_time
        seg = 1 - self.segmentation.astype(np.int8)
        self.stats["low level object voxels"] = np.sum(seg)
        self.stats["low level image voxels"] = np.prod(seg.shape)
        # in seg is now stored low resolution segmentation
        # back to normal parameters
        # step 2: discontinuity localization
        # self.segparams = sparams_hi
        seg_border = scipy.ndimage.filters.laplace(seg, mode="constant")
        logger.debug("seg_border: %s", scipy.stats.describe(seg_border, axis=None))
        # logger.debug(str(np.max(seg_border)))
        # logger.debug(str(np.min(seg_border)))
        seg_border[seg_border != 0] = 1
        logger.debug("seg_border: %s", scipy.stats.describe(seg_border, axis=None))
        # scipy.ndimage.morphology.distance_transform_edt
        boundary_dilatation_distance = self.segparams["boundary_dilatation_distance"]
        seg = scipy.ndimage.morphology.binary_dilation(
            seg_border,
            # seg,
            np.ones(
                [
                    (boundary_dilatation_distance * 2) + 1,
                    (boundary_dilatation_distance * 2) + 1,
                    (boundary_dilatation_distance * 2) + 1,
                ]
            ),
        )
        if self.keep_temp_properties:
            self.temp_msgc_lowres_discontinuity = seg
        else:
            self.temp_msgc_lowres_discontinuity = None

        if self.debug_images:
            import sed3

            pd = sed3.sed3(seg_border)  # ), contour=seg)
            pd.show()
            pd = sed3.sed3(seg)  # ), contour=seg)
            pd.show()
        # segzoom = scipy.ndimage.interpolation.zoom(seg.astype('float'), zoom,
        #                                                order=0).astype('int8')
        self.stats["t3"] = time.time() - start
        return seg
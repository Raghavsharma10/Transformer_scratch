def __hi2lo_multiscale_indexes(self, mask, orig_shape):  # , zoom):
        """
        Function computes multiscale indexes of ndarray.

        mask: Says where is original resolution (0) and where is small
        resolution (1). Mask is in small resolution.

        orig_shape: Original shape of input data.
        zoom: Usually number greater then 1

        result = [[0 1 2],
                  [3 4 4],
                  [5 4 4]]
        """

        mask_orig = zoom_to_shape(mask, orig_shape, dtype=np.int8)

        inds_small = np.arange(mask.size).reshape(mask.shape)
        inds_small_in_orig = zoom_to_shape(inds_small, orig_shape, dtype=np.int8)
        inds_orig = np.arange(np.prod(orig_shape)).reshape(orig_shape)

        # inds_orig = inds_orig * mask_orig
        inds_orig += np.max(inds_small_in_orig) + 1
        # print 'indexes'
        # import py3DSeedEditor as ped
        # import pdb; pdb.set_trace() # BREAKPOINT

        #  '==' is not the same as 'is' for numpy.array
        inds_small_in_orig[mask_orig == True] = inds_orig[mask_orig == True]  # noqa
        inds = inds_small_in_orig
        # print np.max(inds)
        # print np.min(inds)
        inds = relabel_squeeze(inds)
        logger.debug(
            "Index after relabeling: %s", scipy.stats.describe(inds, axis=None)
        )
        # logger.debug("Minimal index after relabeling: " + str(np.min(inds)))
        # inds_orig[mask_orig==True] = 0
        # inds_small_in_orig[mask_orig==False] = 0
        # inds = (inds_orig + np.max(inds_small_in_orig) + 1) + inds_small_in_orig

        return inds, mask_orig
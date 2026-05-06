def _gen_flood_wrap(self, data, rmsimg, innerclip, outerclip=None, domask=False):
        """
        Generator function.
        Segment an image into islands and return one island at a time.

        Needs to work for entire image, and also for components within an island.

        Parameters
        ----------
        data : 2d-array
            Image array.

        rmsimg : 2d-array
            Noise image.

        innerclip, outerclip :float
            Seed (inner) and flood (outer) clipping values.

        domask : bool
            If True then look for a region mask in globals, only return islands that are within the region.
            Default = False.

        Yields
        ------
        data_box : 2d-array
            A island of sources with subthreshold values masked.

        xmin, xmax, ymin, ymax : int
            The corners of the data_box within the initial data array.
        """

        if outerclip is None:
            outerclip = innerclip

        # compute SNR image (data has already been background subtracted)
        snr = abs(data) / rmsimg
        # mask of pixles that are above the outerclip
        a = snr >= outerclip
        # segmentation a la scipy
        l, n = label(a)
        f = find_objects(l)

        if n == 0:
            self.log.debug("There are no pixels above the clipping limit")
            return
        self.log.debug("{1} Found {0} islands total above flood limit".format(n, data.shape))
        # Yield values as before, though they are not sorted by flux
        for i in range(n):
            xmin, xmax = f[i][0].start, f[i][0].stop
            ymin, ymax = f[i][1].start, f[i][1].stop
            if np.any(snr[xmin:xmax, ymin:ymax] > innerclip):  # obey inner clip constraint
                # self.log.info("{1} Island {0} is above the inner clip limit".format(i, data.shape))
                data_box = copy.copy(data[xmin:xmax, ymin:ymax])  # copy so that we don't blank the master data
                data_box[np.where(
                    snr[xmin:xmax, ymin:ymax] < outerclip)] = np.nan  # blank pixels that are outside the outerclip
                data_box[np.where(l[xmin:xmax, ymin:ymax] != i + 1)] = np.nan  # blank out other summits
                # check if there are any pixels left unmasked
                if not np.any(np.isfinite(data_box)):
                    # self.log.info("{1} Island {0} has no non-masked pixels".format(i,data.shape))
                    continue
                if domask and (self.global_data.region is not None):
                    y, x = np.where(snr[xmin:xmax, ymin:ymax] >= outerclip)
                    # convert indices of this sub region to indices in the greater image
                    yx = list(zip(y + ymin, x + xmin))
                    ra, dec = self.global_data.wcshelper.wcs.wcs_pix2world(yx, 1).transpose()
                    mask = self.global_data.region.sky_within(ra, dec, degin=True)
                    # if there are no un-masked pixels within the region then we skip this island.
                    if not np.any(mask):
                        continue
                    self.log.debug("Mask {0}".format(mask))
                # self.log.info("{1} Island {0} will be fit".format(i, data.shape))
                yield data_box, xmin, xmax, ymin, ymax
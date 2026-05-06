def _make_bkg_rms(self, mesh_size=20, forced_rms=None, forced_bkg=None, cores=None):
        """
        Calculate an rms image and a bkg image.

        Parameters
        ----------
        mesh_size : int
            Number of beams per box default = 20

        forced_rms : float
            The rms of the image.
            If None:  calculate the rms level (default).
            Otherwise assume a constant rms.

        forced_bkg : float
            The background level of the image.
            If None: calculate the background level (default).
            Otherwise assume a constant background.

        cores: int
            Number of cores to use if different from what is autodetected.

        """
        if (forced_rms is not None):
            self.log.info("Forcing rms = {0}".format(forced_rms))
            self.global_data.rmsimg[:] = forced_rms
        if (forced_bkg is not None):
            self.log.info("Forcing bkg = {0}".format(forced_bkg))
            self.global_data.bkgimg[:] = forced_bkg

        # If we known both the rms and the bkg then there is nothing to compute
        if (forced_rms is not None) and (forced_bkg is not None):
            return

        data = self.global_data.data_pix
        beam = self.global_data.beam

        img_x, img_y = data.shape
        xcen = int(img_x / 2)
        ycen = int(img_y / 2)

        # calculate a local beam from the center of the data
        pixbeam = self.global_data.psfhelper.get_pixbeam_pixel(xcen, ycen)
        if pixbeam is None:
            self.log.error("Cannot determine the beam shape at the image center")
            sys.exit(1)

        width_x = mesh_size * max(abs(math.cos(np.radians(pixbeam.pa)) * pixbeam.a),
                                  abs(math.sin(np.radians(pixbeam.pa)) * pixbeam.b))
        width_x = int(width_x)
        width_y = mesh_size * max(abs(math.sin(np.radians(pixbeam.pa)) * pixbeam.a),
                                  abs(math.cos(np.radians(pixbeam.pa)) * pixbeam.b))
        width_y = int(width_y)

        self.log.debug("image size x,y:{0},{1}".format(img_x, img_y))
        self.log.debug("beam: {0}".format(beam))
        self.log.debug("mesh width (pix) x,y: {0},{1}".format(width_x, width_y))

        # box centered at image center then tilling outwards
        xstart = int(xcen - width_x / 2) % width_x  # the starting point of the first "full" box
        ystart = int(ycen - width_y / 2) % width_y

        xend = img_x - int(img_x - xstart) % width_x  # the end point of the last "full" box
        yend = img_y - int(img_y - ystart) % width_y

        xmins = [0]
        xmins.extend(list(range(xstart, xend, width_x)))
        xmins.append(xend)

        xmaxs = [xstart]
        xmaxs.extend(list(range(xstart + width_x, xend + 1, width_x)))
        xmaxs.append(img_x)

        ymins = [0]
        ymins.extend(list(range(ystart, yend, width_y)))
        ymins.append(yend)

        ymaxs = [ystart]
        ymaxs.extend(list(range(ystart + width_y, yend + 1, width_y)))
        ymaxs.append(img_y)

        # if the image is smaller than our ideal mesh size, just use the whole image instead
        if width_x >= img_x:
            xmins = [0]
            xmaxs = [img_x]
        if width_y >= img_y:
            ymins = [0]
            ymaxs = [img_y]

        if cores > 1:
            # set up the queue
            queue = pprocess.Queue(limit=cores, reuse=1)
            estimate = queue.manage(pprocess.MakeReusable(self._estimate_bkg_rms))
            # populate the queue
            for xmin, xmax in zip(xmins, xmaxs):
                for ymin, ymax in zip(ymins, ymaxs):
                    estimate(ymin, ymax, xmin, xmax)
        else:
            queue = []
            for xmin, xmax in zip(xmins, xmaxs):
                for ymin, ymax in zip(ymins, ymaxs):
                    queue.append(self._estimate_bkg_rms(xmin, xmax, ymin, ymax))

        # only copy across the bkg/rms if they are not already set
        # queue can only be traversed once so we have to put the if inside the loop
        for ymin, ymax, xmin, xmax, bkg, rms in queue:
            if (forced_rms is None):
                self.global_data.rmsimg[ymin:ymax, xmin:xmax] = rms
            if (forced_rms is None):
                self.global_data.bkgimg[ymin:ymax, xmin:xmax] = bkg

        return
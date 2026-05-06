def result_to_components(self, result, model, island_data, isflags):
        """
        Convert fitting results into a set of components

        Parameters
        ----------
        result : lmfit.MinimizerResult
            The fitting results.

        model : lmfit.Parameters
            The model that was fit.

        island_data : :class:`AegeanTools.models.IslandFittingData`
            Data about the island that was fit.

        isflags : int
            Flags that should be added to this island (in addition to those within the model)

        Returns
        -------
        sources : list
            A list of components, and islands if requested.
        """
        global_data = self.global_data

        # island data
        isle_num = island_data.isle_num
        idata = island_data.i
        xmin, xmax, ymin, ymax = island_data.offsets

        box = slice(int(xmin), int(xmax)), slice(int(ymin), int(ymax))
        rms = global_data.rmsimg[box]
        bkg = global_data.bkgimg[box]
        residual = np.median(result.residual), np.std(result.residual)
        is_flag = isflags

        sources = []
        j = 0
        for j in range(model['components'].value):
            src_flags = is_flag
            source = OutputSource()
            source.island = isle_num
            source.source = j
            self.log.debug(" component {0}".format(j))
            prefix = "c{0}_".format(j)
            xo = model[prefix + 'xo'].value
            yo = model[prefix + 'yo'].value
            sx = model[prefix + 'sx'].value
            sy = model[prefix + 'sy'].value
            theta = model[prefix + 'theta'].value
            amp = model[prefix + 'amp'].value
            src_flags |= model[prefix + 'flags'].value

            # these are goodness of fit statistics for the entire island.
            source.residual_mean = residual[0]
            source.residual_std = residual[1]
            # set the flags
            source.flags = src_flags

            # #pixel pos within island +
            # island offset within region +
            # region offset within image +
            # 1 for luck
            # (pyfits->fits conversion = luck)
            x_pix = xo + xmin + 1
            y_pix = yo + ymin + 1
            # update the source xo/yo so the error calculations can be done correctly
            # Note that you have to update the max or the value you set will be clipped at the max allowed value
            model[prefix + 'xo'].set(value=x_pix, max=np.inf)
            model[prefix + 'yo'].set(value=y_pix, max=np.inf)
            # ------ extract source parameters ------

            # fluxes
            # the background is taken from background map
            # Clamp the pixel location to the edge of the background map
            y = max(min(int(round(y_pix - ymin)), bkg.shape[1] - 1), 0)
            x = max(min(int(round(x_pix - xmin)), bkg.shape[0] - 1), 0)
            source.background = bkg[x, y]
            source.local_rms = rms[x, y]
            source.peak_flux = amp

            # all params are in degrees
            source.ra, source.dec, source.a, source.b, source.pa = global_data.wcshelper.pix2sky_ellipse((x_pix, y_pix),
                                                                                                         sx * CC2FHWM,
                                                                                                         sy * CC2FHWM,
                                                                                                         theta)
            source.a *= 3600  # arcseconds
            source.b *= 3600
            # force a>=b
            fix_shape(source)
            # limit the pa to be in (-90,90]
            source.pa = pa_limit(source.pa)

            # if one of these values are nan then there has been some problem with the WCS handling
            if not all(np.isfinite((source.ra, source.dec, source.a, source.b, source.pa))):
                src_flags |= flags.WCSERR
            # negative degrees is valid for RA, but I don't want them.
            if source.ra < 0:
                source.ra += 360
            source.ra_str = dec2hms(source.ra)
            source.dec_str = dec2dms(source.dec)

            # calculate integrated flux
            source.int_flux = source.peak_flux * sx * sy * CC2FHWM ** 2 * np.pi
            # scale Jy/beam -> Jy using the area of the beam
            source.int_flux /= global_data.psfhelper.get_beamarea_pix(source.ra, source.dec)

            # Calculate errors for params that were fit (as well as int_flux)
            errors(source, model, global_data.wcshelper)

            source.flags = src_flags
            # add psf info
            local_beam = global_data.psfhelper.get_beam(source.ra, source.dec)
            if local_beam is not None:
                source.psf_a = local_beam.a * 3600
                source.psf_b = local_beam.b * 3600
                source.psf_pa = local_beam.pa
            else:
                source.psf_a = 0
                source.psf_b = 0
                source.psf_pa = 0
            sources.append(source)
            self.log.debug(source)

        if global_data.blank:
            outerclip = island_data.scalars[1]
            idx, idy = np.where(abs(idata) - outerclip * rms > 0)
            idx += xmin
            idy += ymin
            self.global_data.img._pixels[[idx, idy]] = np.nan

        # calculate the integrated island flux if required
        if island_data.doislandflux:
            _, outerclip, _ = island_data.scalars
            self.log.debug("Integrated flux for island {0}".format(isle_num))
            kappa_sigma = np.where(abs(idata) - outerclip * rms > 0, idata, np.NaN)
            self.log.debug("- island shape is {0}".format(kappa_sigma.shape))

            source = IslandSource()
            source.flags = 0
            source.island = isle_num
            source.components = j + 1
            source.peak_flux = np.nanmax(kappa_sigma)
            # check for negative islands
            if source.peak_flux < 0:
                source.peak_flux = np.nanmin(kappa_sigma)
            self.log.debug("- peak flux {0}".format(source.peak_flux))

            # positions and background
            if np.isfinite(source.peak_flux):
                positions = np.where(kappa_sigma == source.peak_flux)
            else:  # if a component has been refit then it might have flux = np.nan
                positions = [[kappa_sigma.shape[0] / 2], [kappa_sigma.shape[1] / 2]]
            xy = positions[0][0] + xmin, positions[1][0] + ymin
            radec = global_data.wcshelper.pix2sky(xy)
            source.ra = radec[0]

            # convert negative ra's to positive ones
            if source.ra < 0:
                source.ra += 360

            source.dec = radec[1]
            source.ra_str = dec2hms(source.ra)
            source.dec_str = dec2dms(source.dec)
            source.background = bkg[positions[0][0], positions[1][0]]
            source.local_rms = rms[positions[0][0], positions[1][0]]
            source.x_width, source.y_width = idata.shape
            source.pixels = int(sum(np.isfinite(kappa_sigma).ravel() * 1.0))
            source.extent = [xmin, xmax, ymin, ymax]

            # TODO: investigate what happens when the sky coords are skewed w.r.t the pixel coords
            # calculate the area of the island as a fraction of the area of the bounding box
            bl = global_data.wcshelper.pix2sky([xmax, ymin])
            tl = global_data.wcshelper.pix2sky([xmax, ymax])
            tr = global_data.wcshelper.pix2sky([xmin, ymax])
            height = gcd(tl[0], tl[1], bl[0], bl[1])
            width = gcd(tl[0], tl[1], tr[0], tr[1])
            area = height * width
            source.area = area * source.pixels / source.x_width / source.y_width  # area is in deg^2

            # create contours
            msq = MarchingSquares(idata)
            source.contour = [(a[0] + xmin, a[1] + ymin) for a in msq.perimeter]
            # calculate the maximum angular size of this island, brute force method
            source.max_angular_size = 0
            for i, pos1 in enumerate(source.contour):
                radec1 = global_data.wcshelper.pix2sky(pos1)
                for j, pos2 in enumerate(source.contour[i:]):
                    radec2 = global_data.wcshelper.pix2sky(pos2)
                    dist = gcd(radec1[0], radec1[1], radec2[0], radec2[1])
                    if dist > source.max_angular_size:
                        source.max_angular_size = dist
                        source.pa = bear(radec1[0], radec1[1], radec2[0], radec2[1])
                        source.max_angular_size_anchors = [pos1[0], pos1[1], pos2[0], pos2[1]]

            self.log.debug("- peak position {0}, {1} [{2},{3}]".format(source.ra_str, source.dec_str, positions[0][0],
                                                                       positions[1][0]))

            # integrated flux
            beam_area = global_data.psfhelper.get_beamarea_deg2(source.ra, source.dec)  # beam in deg^2
            # get_beamarea_pix(source.ra, source.dec)  # beam is in pix^2
            isize = source.pixels  # number of non zero pixels
            self.log.debug("- pixels used {0}".format(isize))
            source.int_flux = np.nansum(kappa_sigma)  # total flux Jy/beam
            self.log.debug("- sum of pixles {0}".format(source.int_flux))
            source.int_flux *= beam_area  # total flux in Jy
            self.log.debug("- integrated flux {0}".format(source.int_flux))
            eta = erf(np.sqrt(-1 * np.log(abs(source.local_rms * outerclip / source.peak_flux)))) ** 2
            self.log.debug("- eta {0}".format(eta))
            source.eta = eta
            source.beam_area = beam_area

            # I don't know how to calculate this error so we'll set it to nan
            source.err_int_flux = np.nan
            sources.append(source)
        return sources
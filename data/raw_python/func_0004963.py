def radial_average(self, qrange=None, pixel=False, returnmask=False,
                       errorpropagation=3, abscissa_errorpropagation=3,
                       raw_result=False) -> Curve:
        """Do a radial averaging

        Inputs:
            qrange: the q-range. If None, auto-determine. If 'linear', auto-determine
                with linear spacing (same as None). If 'log', auto-determine
                with log10 spacing.
            pixel: do a pixel-integration (instead of q)
            returnmask: if the effective mask matrix is to be returned.
            errorpropagation: the type of error propagation (3: highest of squared or
                std-dev, 2: squared, 1: linear, 0: independent measurements of
                the same quantity)
            abscissa_errorpropagation: the type of the error propagation in the
                abscissa (3: highest of squared or std-dev, 2: squared, 1: linear,
                0: independent measurements of the same quantity)
            raw_result: if True, do not pack the result in a SASCurve, return the
                individual np.ndarrays.

        Outputs:
            the one-dimensional curve as an instance of SASCurve (if pixel is
                False) or SASPixelCurve (if pixel is True), if raw_result was True.
                otherwise the q (or pixel), dq (or dpixel), I, dI, area vectors
            the mask matrix (if returnmask was True)
        """
        retmask = None
        if isinstance(qrange, str):
            if qrange == 'linear':
                qrange = None
                autoqrange_linear = True
            elif qrange == 'log':
                qrange = None
                autoqrange_linear = False
            else:
                raise ValueError(
                        'Value given for qrange (''%s'') not understood.' % qrange)
        else:
            autoqrange_linear = True  # whatever
        if pixel:
            abscissa_kind = 3
        else:
            abscissa_kind = 0
        res = radint_fullq_errorprop(self.intensity, self.error, self.header.wavelength.val,
                                     self.header.wavelength.err, self.header.distance.val,
                                     self.header.distance.err, self.header.pixelsizey.val,
                                     self.header.pixelsizex.val, self.header.beamcentery.val,
                                     self.header.beamcentery.err, self.header.beamcenterx.val,
                                     self.header.beamcenterx.err, (self.mask == 0).astype(np.uint8),
                                     qrange, returnmask=returnmask, errorpropagation=errorpropagation,
                                     autoqrange_linear=autoqrange_linear, abscissa_kind=abscissa_kind,
                                     abscissa_errorpropagation=abscissa_errorpropagation)
        q, dq, I, E, area = res[:5]
        if not raw_result:
            c = Curve(q, I, E, dq)
            if returnmask:
                return c, res[5]
            else:
                return c
        else:
            if returnmask:
                return q, dq, I, E, area, res[5]
            else:
                return q, dq, I, E, area
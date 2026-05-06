def standardUncertainties(self, focal_Length_mm, f_number, midpointdepth=1000,
                              focusAtYX=None,
                              # sigma_best_focus=0,
                              # quad_pos_err=0,
                              shape=None,
                              uncertainties=(0, 0)):
        '''
        focusAtXY - image position with is in focus
            if not set it is assumed that the image middle is in focus
        sigma_best_focus - standard deviation of the PSF
                             within the best focus (default blur)
        uncertainties - contibutors for standard uncertainty
                        these need to be perspective transformed to fit the new 
                        image shape
        '''
        # TODO: consider quad_pos_error
        # (also influences intensity corr map)

        if shape is None:
            s = self.img.shape
        else:
            s = shape

        # 1. DEFOCUS DUE TO DEPTH OF FIELD
        ##################################
        depthMap = self.depthMap(midpointdepth)
        if focusAtYX is None:
            # assume image middle is in-focus:
            focusAtYX = s[0] // 2, s[1] // 2
        infocusDepth = depthMap[focusAtYX]
        depthOfField_blur = defocusThroughDepth(
            depthMap, infocusDepth, focal_Length_mm, f_number, k=2.335)

        # 2. INCREAASED PIXEL SIZE DUE TO INTERPOLATION BETWEEN
        #   PIXELS MOVED APARD
        ######################################################
        # index maps:
        py, px = np.mgrid[0:s[0], 0:s[1]]
        # warped index maps:
        wx = cv2.warpPerspective(np.asfarray(px), self.homography,
                                 self._newBorders,
                                 borderValue=np.nan,
                                 flags=cv2.INTER_LANCZOS4)
        wy = cv2.warpPerspective(np.asfarray(py), self.homography,
                                 self._newBorders,
                                 borderValue=np.nan,
                                 flags=cv2.INTER_LANCZOS4)

        pxSizeFactorX = 1 / np.abs(np.gradient(wx)[1])
        pxSizeFactorY = 1 / np.abs(np.gradient(wy)[0])

        # WARP ALL FIELD TO NEW PERSPECTIVE AND MULTIPLY WITH PXSIZE FACTOR:
        depthOfField_blur = cv2.warpPerspective(
            depthOfField_blur, self.homography, self._newBorders,
            borderValue=np.nan,
        )

        # perspective transform given uncertainties:
        warpedU = []
        for u in uncertainties:
            #             warpedU.append([])
            #             for i in u:
            # print i, type(i), isinstance(i, np.ndarray)
            if isinstance(u, np.ndarray) and u.size > 1:
                u = cv2.warpPerspective(u, self.homography,
                                        self._newBorders,
                                        borderValue=np.nan,
                                        flags=cv2.INTER_LANCZOS4)  # *f

            else:
                # multiply with area ratio: after/before perspective warp
                u *= self.areaRatio

            warpedU.append(u)

        # given uncertainties after warp:
        ux, uy = warpedU

        ux = pxSizeFactorX * (ux**2 + depthOfField_blur**2)**0.5
        uy = pxSizeFactorY * (uy**2 + depthOfField_blur**2)**0.5

        # TODO: remove depthOfField_blur,fx,fy from return
        return ux, uy, depthOfField_blur, pxSizeFactorX, pxSizeFactorY
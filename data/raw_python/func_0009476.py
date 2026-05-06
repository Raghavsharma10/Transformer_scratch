def addImg(self, img, maxShear=0.015, maxRot=100, minMatches=12,
               borderWidth=3):  # borderWidth=100
        """
        Args:
            img (path or array): image containing the same object as in the reference image
        Kwargs:
            maxShear (float): In order to define a good fit, refect higher shear values between
                              this and the reference image
            maxRot (float): Same for rotation
            minMatches (int): Minimum of mating points found in both, this and the reference image
        """
        try:
            fit, img, H, H_inv, nmatched = self._fitImg(img)
        except Exception as e:
            print(e)
            return

        # CHECK WHETHER FIT IS GOOD ENOUGH:
        (translation, rotation, scale, shear) = decompHomography(H)
        print('Homography ...\n\ttranslation: %s\n\trotation: %s\n\tscale: %s\n\tshear: %s'
              % (translation, rotation, scale, shear))
        if (nmatched > minMatches
                and abs(shear) < maxShear
                and abs(rotation) < maxRot):
            print('==> img added')
            # HOMOGRAPHY:
            self.Hs.append(H)
            # INVERSE HOMOGRSAPHY
            self.Hinvs.append(H_inv)
            # IMAGES WARPED TO THE BASE IMAGE
            self.fits.append(fit)
            # ADD IMAGE TO THE INITIAL flatField ARRAY:
            i = img > self.signal_ranges[-1][0]

            # remove borders (that might have erroneous light):
            i = minimum_filter(i, borderWidth)

            self._ff_mma.update(img, i)

            # create fit img mask:
            mask = fit < self.signal_ranges[-1][0]
            mask = maximum_filter(mask, borderWidth)
            # IGNORE BORDER
            r = self.remove_border_size
            if r:
                mask[:r, :] = 1
                mask[-r:, :] = 1
                mask[:, -r:] = 1
                mask[:, :r] = 1
            self._fit_masks.append(mask)

            # image added
            return fit
        return False
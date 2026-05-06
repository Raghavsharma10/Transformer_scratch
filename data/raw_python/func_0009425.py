def _findOverlap(self, img_rgb, overlap, overlapDeviation,
                     rotation, rotationDeviation):
        '''
        return offset(x,y) which fit best self._base_img
        through template matching
        '''
        # get gray images
        if len(img_rgb.shape) != len(img_rgb.shape):
            raise Exception(
                'number of channels(colors) for both images different')
        if overlapDeviation == 0 and rotationDeviation == 0:
            return (0, overlap, rotation)

        s = self.base_img_rgb.shape
        ho = int(round(overlap * 0.5))
        overlap = int(round(overlap))
        # create two image cuts to compare:
        imgcut = self.base_img_rgb[s[0] - overlapDeviation - overlap:, :]
        template = img_rgb[:overlap, ho:s[1] - ho]

        def fn(angle):
            rotTempl = self._rotate(template, angle)
            # Apply template Matching
            fn.res = cv2.matchTemplate(rotTempl.astype(np.float32),
                                       imgcut.astype(np.float32),
                                       cv2.TM_CCORR_NORMED)
            return 1 / fn.res.mean()

        if rotationDeviation == 0:
            angle = rotation
            fn(rotation)

        else:
            # find best rotation angle:
            angle = brent(fn, brack=(rotation - rotationDeviation,
                                     rotation + rotationDeviation))

        loc = cv2.minMaxLoc(fn.res)[-1]

        offsx = int(round(loc[0] - ho))
        offsy = overlapDeviation + overlap - loc[1]
        return offsx, offsy, angle
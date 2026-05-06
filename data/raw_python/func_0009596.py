def correct(self, img):
        '''
        ...from perspective distortion:
         --> perspective transformation
         --> apply tilt factor (view factor) correction 
        '''
        print("CORRECT PERSPECTIVE ...")
        self.img = imread(img)

        if not self._homography_is_fixed:
            self._homography = None
        h = self.homography

        if self.opts['do_correctIntensity']:
            tf = self.tiltFactor()
            self.img = np.asfarray(self.img)
            if self.img.ndim == 3:
                for col in range(self.img.shape[2]):
                    self.img[..., col] /= tf
            else:
                self.img = self.img / tf
        warped = cv2.warpPerspective(self.img,
                                     h,
                                     self._newBorders[::-1],
                                     flags=cv2.INTER_LANCZOS4,
                                     **self.opts['cv2_opts'])
        return warped
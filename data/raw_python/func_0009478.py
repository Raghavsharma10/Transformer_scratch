def _fitImg(self, img):
        '''
        fit perspective and size of the input image to the reference image
        '''
        img = imread(img, 'gray')
        if self.bg is not None:
            img = cv2.subtract(img, self.bg)

        if self.lens is not None:
            img = self.lens.correct(img, keepSize=True)

        (H, _, _, _, _, _, _, n_matches) = self.findHomography(img)
        H_inv = self.invertHomography(H)

        s = self.obj_shape
        fit = cv2.warpPerspective(img, H_inv, (s[1], s[0]))
        return fit, img, H, H_inv, n_matches
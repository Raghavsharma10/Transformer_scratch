def fitImg(self, img_rgb):
        '''
        fit perspective and size of the input image to the base image
        '''
        H = self.pattern.findHomography(img_rgb)[0]
        H_inv = self.pattern.invertHomography(H)
        s = self.img_orig.shape
        warped = cv2.warpPerspective(img_rgb, H_inv, (s[1], s[0]))
        return warped
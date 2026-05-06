def correct(self, image, keepSize=False, borderValue=0):
        '''
        remove lens distortion from given image
        '''
        image = imread(image)
        (h, w) = image.shape[:2]
        mapx, mapy = self.getUndistortRectifyMap(w, h)
        self.img = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=borderValue
                             )
        if not keepSize:
            xx, yy, ww, hh = self.roi
            self.img = self.img[yy: yy + hh, xx: xx + ww]
        return self.img
def _rotate(img, angle):
        '''
        angle [DEG]
        '''
        s = img.shape
        if angle == 0:
            return img
        else:
            M = cv2.getRotationMatrix2D((s[1] // 2,
                                         s[0] // 2), angle, 1)
            return cv2.warpAffine(img, M, (s[1], s[0]))